"""Run this first to find the device index for your mic and headphones
(e.g. AirPods Max), then set them in config.yaml under audio.input_device /
audio.output_device."""
import sounddevice as sd

print(sd.query_devices())
print("\nDefault input:", sd.default.device[0], "| Default output:", sd.default.device[1])
