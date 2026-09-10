"""
Shared pipeline code: device resolution, the SED-based trigger detector, and
the Suppressor class that owns the live mic-in -> detect -> mute -> play-out
loop as start()/stop()-able background threads.

Used by both main.py (CLI) and app.py (local web dashboard).
"""
import collections
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(spec, kind):
    """Resolve an audio device spec from config to a device index, fresh,
    every time this is called. macOS reshuffles CoreAudio device indices
    whenever a Bluetooth device connects/disconnects, so numeric indices
    saved in config.yaml go stale -- resolving by name substring instead is
    much more robust. `kind` is 'input' or 'output'.

    spec may be:
      - None -> system default for that kind
      - an int -> used as-is (escape hatch, but will go stale)
      - a string -> matched against device names, filtered to devices that
        actually support the requested direction (since e.g. "AirPods Max"
        appears twice in the device list, once per direction)
    """
    if spec is None:
        return None
    if isinstance(spec, int):
        return spec

    devices = sd.query_devices()
    chan_key = "max_input_channels" if kind == "input" else "max_output_channels"
    matches = [
        i for i, d in enumerate(devices)
        if spec.lower() in d["name"].lower() and d[chan_key] > 0
    ]
    if not matches:
        available = [d["name"] for d in devices if d[chan_key] > 0]
        raise RuntimeError(
            f"No {kind} device matching '{spec}' is currently connected. "
            f"Available {kind} devices right now: {available}. "
            f"(If this is a Bluetooth device like AirPods, make sure it's "
            f"powered on and connected.)"
        )
    if len(matches) > 1:
        print(f"Warning: multiple {kind} devices match '{spec}': "
              f"{[devices[i]['name'] for i in matches]}. Using the first.",
              file=sys.stderr)
    return matches[0]


def normalize_for_model(audio):
    """PANNs was trained on audio in [-1, 1]. If the input device is hot
    (peak > 1.0, i.e. clipping/overs at the driver level), scale it back down
    so we at least feed the model something in-range, rather than silently
    handing it distorted/out-of-distribution samples."""
    peak = np.abs(audio).max()
    if peak > 1.0:
        return audio / peak
    return audio


def resolve_class_indices(cfg, labels):
    """Map each trigger's configured class names to indices in the model's
    label list. Fails fast with suggestions if a name doesn't match exactly."""
    import difflib

    label_to_idx = {name: i for i, name in enumerate(labels)}
    resolved = {}
    for trigger_name, trigger_cfg in cfg["triggers"].items():
        idxs = []
        for class_name in trigger_cfg["classes"]:
            if class_name not in label_to_idx:
                close = difflib.get_close_matches(class_name, labels, n=3)
                raise ValueError(
                    f"Class '{class_name}' (trigger '{trigger_name}') not found "
                    f"in model labels. Closest matches: {close}"
                )
            idxs.append(label_to_idx[class_name])
        resolved[trigger_name] = {
            "indices": idxs,
            "threshold": trigger_cfg["threshold"],
        }
    return resolved


class GainSmoother:
    """Carries gain state across blocks and ramps toward a target gain at a
    bounded rate (attack when muting, release when unmuting), so we never
    step the volume abruptly (which causes audible clicks)."""

    def __init__(self, sample_rate, attack_ms, release_ms):
        self.gain = 1.0
        self.sr = sample_rate
        self.attack_per_sample = 1.0 / (attack_ms / 1000.0 * sample_rate)
        self.release_per_sample = 1.0 / (release_ms / 1000.0 * sample_rate)

    def process(self, n_samples, target_gain):
        """Return an (n_samples,) ramp from current gain toward target_gain,
        updating internal state to the ramp's final value."""
        out = np.empty(n_samples, dtype=np.float32)
        g = self.gain
        if target_gain < g:
            step = -self.attack_per_sample   # muting: fast-ish fade down
        else:
            step = self.release_per_sample    # unmuting: fade up
        for i in range(n_samples):
            if abs(target_gain - g) < abs(step):
                g = target_gain
            else:
                g += step
            out[i] = g
        self.gain = g
        return out


class TriggerDetector:
    def __init__(self, cfg, device="cpu"):
        from panns_inference import SoundEventDetection, labels

        self.cfg = cfg
        self.sr = cfg["audio"]["sample_rate"]
        self.window_s = cfg["audio"]["window_s"]
        self.fps = 100  # PANNs framewise output rate (10ms/frame at sr=32000)

        print(f"Loading PANNs SED model on device={device} (first run downloads "
              f"~300MB checkpoint, be patient)...", file=sys.stderr)
        self.sed = SoundEventDetection(checkpoint_path=None, device=device)
        self.class_map = resolve_class_indices(cfg, labels)
        print(f"Resolved triggers: "
              f"{ {k: cfg['triggers'][k]['classes'] for k in self.class_map} }",
              file=sys.stderr)

        self.context = np.zeros(int(self.window_s * self.sr), dtype=np.float32)
        self.hold_ms = cfg["audio"]["hold_ms"]
        self.last_trigger_t = -1e9
        self.last_infer_ms = None

    def push_block_and_detect(self, block, now_t):
        """Append `block` (mono float32 @ self.sr) to the rolling context,
        run inference, and return (muted: bool, which_trigger: str|None)."""
        n = len(block)
        self.context = np.roll(self.context, -n)
        self.context[-n:] = block

        t0 = time.perf_counter()
        ctx_for_model = normalize_for_model(self.context)
        framewise = self.sed.inference(ctx_for_model[None, :])[0]  # (frames, classes)
        self.last_infer_ms = (time.perf_counter() - t0) * 1000

        block_frames = max(1, int(round(n / self.sr * self.fps)))
        tail = framewise[-block_frames:]  # frames corresponding to the new block

        fired = None
        for name, info in self.class_map.items():
            prob = tail[:, info["indices"]].max()
            if prob >= info["threshold"]:
                fired = name
                break

        if fired is not None:
            self.last_trigger_t = now_t

        muted = (now_t - self.last_trigger_t) * 1000.0 < self.hold_ms
        return muted, fired


class Suppressor:
    """Owns the live audio pipeline as start()/stop()-able background
    threads. The model is loaded once at construction (expensive, ~seconds);
    start()/stop() just open/close audio streams (cheap), so this is safe to
    toggle repeatedly from a UI."""

    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.detector = TriggerDetector(cfg, device=device)
        self.sr = cfg["audio"]["sample_rate"]
        self.block_samples = int(cfg["audio"]["block_ms"] / 1000.0 * self.sr)
        self.delay_blocks = max(1, int(round(
            cfg["audio"]["delay_s"] * 1000 / cfg["audio"]["block_ms"])))

        self._lifecycle_lock = threading.Lock()
        self._running = False
        self._in_stream = None
        self._out_stream = None
        self._worker_thread = None
        self._stop_flag = None
        self._input_q = None
        self._output_q = None
        self._output_lock = None

        self.in_device_name = None
        self.out_device_name = None
        self.events = collections.deque(maxlen=50)  # newest first

    def is_running(self):
        return self._running

    def start(self):
        with self._lifecycle_lock:
            if self._running:
                return
            try:
                in_dev = resolve_device(self.cfg["audio"]["input_device"], "input")
                out_dev = resolve_device(self.cfg["audio"]["output_device"], "output")
                self.in_device_name = (
                    sd.query_devices(in_dev)["name"] if in_dev is not None else "(default)")
                self.out_device_name = (
                    sd.query_devices(out_dev)["name"] if out_dev is not None else "(default)")

                self._input_q = queue.Queue()
                self._output_q = collections.deque()
                self._output_lock = threading.Lock()
                self._stop_flag = threading.Event()
                smoother = GainSmoother(self.sr, self.cfg["audio"]["attack_ms"],
                                         self.cfg["audio"]["release_ms"])

                def in_callback(indata, frames, time_info, status):
                    self._input_q.put(indata[:, 0].copy())

                def worker():
                    while not self._stop_flag.is_set():
                        try:
                            block = self._input_q.get(timeout=0.5)
                        except queue.Empty:
                            continue
                        muted, fired = self.detector.push_block_and_detect(block, time.time())
                        target_gain = 0.0 if muted else 1.0
                        gain_ramp = smoother.process(len(block), target_gain)
                        processed = block * gain_ramp
                        if fired:
                            self.events.appendleft({"time": time.time(), "trigger": fired})
                            print(f"[{time.strftime('%H:%M:%S')}] trigger detected: {fired}",
                                  file=sys.stderr)
                        with self._output_lock:
                            self._output_q.append(processed.astype(np.float32))

                def out_callback(outdata, frames, time_info, status):
                    with self._output_lock:
                        if len(self._output_q) == 0:
                            outdata[:, 0] = 0
                            return
                        chunk = self._output_q.popleft()
                    if len(chunk) < frames:
                        chunk = np.pad(chunk, (0, frames - len(chunk)))
                    elif len(chunk) > frames:
                        with self._output_lock:
                            self._output_q.appendleft(chunk[frames:])
                        chunk = chunk[:frames]
                    outdata[:, 0] = chunk

                self._worker_thread = threading.Thread(target=worker, daemon=True)
                self._worker_thread.start()

                self._in_stream = sd.InputStream(
                    samplerate=self.sr, blocksize=self.block_samples, channels=1,
                    dtype="float32", device=in_dev, callback=in_callback)
                self._in_stream.start()

                deadline = time.time() + 10
                while len(self._output_q) < self.delay_blocks and time.time() < deadline:
                    time.sleep(0.05)

                self._out_stream = sd.OutputStream(
                    samplerate=self.sr, blocksize=self.block_samples, channels=1,
                    dtype="float32", device=out_dev, callback=out_callback)
                self._out_stream.start()

                self._running = True
            except Exception:
                self._teardown()
                raise

    def stop(self):
        with self._lifecycle_lock:
            if not self._running:
                return
            self._teardown()

    def _teardown(self):
        if self._stop_flag:
            self._stop_flag.set()
        for stream in (self._in_stream, self._out_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2)
        self._in_stream = None
        self._out_stream = None
        self._worker_thread = None
        self._running = False

    def set_threshold(self, trigger_name, value):
        if trigger_name not in self.detector.class_map:
            raise KeyError(trigger_name)
        self.detector.class_map[trigger_name]["threshold"] = float(value)

    def status(self):
        recent = list(self.events)[:15]
        return {
            "running": self._running,
            "input_device": self.in_device_name,
            "output_device": self.out_device_name,
            "recent_events": recent,
            "last_trigger": recent[0]["trigger"] if recent else None,
            "last_trigger_time": recent[0]["time"] if recent else None,
            "thresholds": {
                name: info["threshold"] for name, info in self.detector.class_map.items()
            },
            "classes": {
                name: self.cfg["triggers"][name]["classes"]
                for name in self.detector.class_map
            },
        }
