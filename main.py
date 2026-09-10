"""
Live misophonia trigger suppressor (CLI).

Captures your mic, runs a pretrained sound-event-detection model (PANNs/CNN14)
on a rolling window to spot trigger sounds (chewing, tapping, throat clearing,
heavy breathing, crunching, etc. -- see config.yaml), and plays the audio back
through your headphones with those moments smoothly muted.

For a local web dashboard instead of the terminal, use `python3 app.py`.

Run `python list_devices.py` first if you need to check device names.
"""
import argparse
import time

import numpy as np

from suppressor import Suppressor, TriggerDetector, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                     help="torch device for inference")
    ap.add_argument("--benchmark", action="store_true",
                     help="Just measure inference latency and exit (no audio I/O)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.benchmark:
        detector = TriggerDetector(cfg, device=args.device)
        block_samples = int(cfg["audio"]["block_ms"] / 1000.0 * cfg["audio"]["sample_rate"])
        dummy = np.random.randn(block_samples).astype(np.float32) * 0.01
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            detector.push_block_and_detect(dummy, time.time())
            times.append((time.perf_counter() - t0) * 1000)
        print(f"Inference latency over {len(times)} runs: "
              f"mean={np.mean(times):.1f}ms  max={np.max(times):.1f}ms  "
              f"block_ms={cfg['audio']['block_ms']}")
        if np.mean(times) > cfg["audio"]["block_ms"]:
            print("WARNING: mean inference time exceeds block_ms -- the "
                  "pipeline will fall behind real time. Increase block_ms "
                  "and/or window_s in config.yaml, or try --device mps.")
        return

    sup = Suppressor(cfg, device=args.device)
    sup.start()
    st = sup.status()
    print(f"Resolved input device: {st['input_device']}")
    print(f"Resolved output device: {st['output_device']}")
    print("Running. Ctrl+C to stop.")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        sup.stop()


if __name__ == "__main__":
    main()
