"""Diagnostic: records a few seconds of audio from your mic, runs it through
the same SED model as main.py, and prints out the actual confidence scores
over time for each configured trigger class. Use this to see whether a
trigger is being detected at all (and how strong), vs just falling under
threshold.

Usage:
    python3 debug_listen.py --seconds 6
    (then type on your keyboard / make the trigger sound during the recording)
"""
import argparse
import sys

import numpy as np
import sounddevice as sd
import yaml

from suppressor import TriggerDetector, load_config, normalize_for_model, resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sr = cfg["audio"]["sample_rate"]
    in_dev = resolve_device(cfg["audio"]["input_device"], "input")
    print(f"Resolved input device: "
          f"{sd.query_devices(in_dev)['name'] if in_dev is not None else '(default)'}",
          file=sys.stderr)

    detector = TriggerDetector(cfg, device=args.device)

    print(f"\nRecording {args.seconds}s from device={in_dev} (default if None) "
          f"-- MAKE YOUR TRIGGER SOUND NOW", file=sys.stderr)
    audio = sd.rec(int(args.seconds * sr), samplerate=sr, channels=1,
                    dtype="float32", device=in_dev)
    sd.wait()
    audio = audio[:, 0]
    peak = np.abs(audio).max()
    print(f"Recorded. Peak level: {peak:.4f} "
          f"(if this is near 0.0, the mic isn't picking up sound -- check "
          f"device/volume; if this is above 1.0, your input is clipping -- "
          f"lower the mic input level in System Settings > Sound > Input)",
          file=sys.stderr)

    from panns_inference import labels
    label_to_idx = {name: i for i, name in enumerate(labels)}

    framewise = detector.sed.inference(normalize_for_model(audio)[None, :])[0]  # (frames, classes)
    n_frames = framewise.shape[0]
    fps = n_frames / args.seconds

    print(f"\nframes={n_frames}  fps~{fps:.1f}\n")
    print(f"{'trigger':<18} {'classes':<40} {'max_prob':>9} {'threshold':>10} {'would_fire':>11}")
    for trigger_name, trigger_cfg in cfg["triggers"].items():
        idxs = [label_to_idx[c] for c in trigger_cfg["classes"]]
        probs_over_time = framewise[:, idxs].max(axis=1)  # per-frame max over this trigger's classes
        max_prob = probs_over_time.max()
        threshold = trigger_cfg["threshold"]
        fires = max_prob >= threshold
        print(f"{trigger_name:<18} {str(trigger_cfg['classes']):<40} "
              f"{max_prob:>9.3f} {threshold:>10.3f} {str(fires):>11}")

    print(f"\nTop 10 classes overall by peak probability during this recording:")
    peak_by_class = framewise.max(axis=0)
    top_idx = np.argsort(peak_by_class)[::-1][:10]
    for i in top_idx:
        print(f"  {labels[i]:<40} {peak_by_class[i]:.3f}")


if __name__ == "__main__":
    main()
