# Misophonia Trigger Suppressor

Live mic → sound detection → smooth mute → headphone playback. Runs entirely
locally on your Mac: it listens through your built-in mic, uses a pretrained
sound-event-detection model to spot misophonia trigger sounds in real time,
and plays the audio back through your headphones with just those moments
muted — everything else passes through untouched.

No training, no cloud calls. Detection is done by
[PANNs (Cnn14)](https://github.com/qiuqiangkong/audioset_tagging_cnn), a model
pretrained on Google's AudioSet taxonomy (527 sound classes), run locally on
CPU.

## Configured triggers

Defined in `config.yaml`, each mapped to the closest AudioSet class(es):

| Trigger | AudioSet class(es) |
|---|---|
| `chewing` | Chewing, mastication |
| `crunching` | Crunch |
| `tapping_on_phone` | Tap, Clicking |
| `keyboard_typing` | Typing, Computer keyboard |
| `throat_clearing` | Throat clearing |
| `heavy_breathing` | Breathing |

`tapping_on_phone` and `heavy_breathing` don't have an exact AudioSet match
(no "phone tap" or "heavy" vs. "light" breathing distinction exists), so
they're mapped to the closest general class and may need threshold tuning to
avoid over/under-firing.

## Setup

```bash
git clone https://github.com/katrinat-23/misophonia-suppressor.git
cd misophonia-suppressor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The first time you run anything that loads the model, `panns_inference` needs
two files that it normally fetches with `wget` — if you don't have `wget`
installed, grab them with `curl` instead:

```bash
mkdir -p ~/panns_data
curl -sL -o ~/panns_data/class_labels_indices.csv \
  "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
curl -L -o ~/panns_data/Cnn14_DecisionLevelMax.pth \
  "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
```

The checkpoint is ~327MB — confirm the download finished fully (`ls -la
~/panns_data/`) before running anything; a truncated file fails with a
cryptic `RuntimeError: unexpected EOF` from `torch.load`.

## Configure your audio devices

Check what's available:

```bash
python3 list_devices.py
```

Then set `audio.input_device` / `audio.output_device` in `config.yaml` to
**name substrings** (not numeric indices — see "Device index drift" below),
e.g.:

```yaml
audio:
  input_device: "MacBook Pro Microphone"
  output_device: "AirPods Max"
```

**Use headphones for output, not speakers** — if the mic can hear your own
speaker output, you get a feedback loop.

Input and output devices don't have to be the same. In practice, a
Bluetooth headset's own mic (e.g. AirPods) often applies aggressive
voice-isolation DSP that scrubs out non-speech sound before your code ever
sees it — using the laptop's built-in mic for input while keeping the
headphones for output avoids that.

## Running it

### Web dashboard (recommended)

```bash
python3 app.py
```

Open **http://127.0.0.1:5757**. Start/Stop the pipeline, watch live
detections, and drag per-trigger sensitivity sliders — changes apply
immediately, no restart needed.

### Terminal

```bash
python3 main.py
```

Prints a line every time it mutes something. `Ctrl+C` to stop.

### Diagnostics

If a trigger doesn't seem to fire, don't just guess at thresholds — check the
model's actual confidence:

```bash
python3 debug_listen.py --seconds 6
```

Make the trigger sound when it says to, then check the printed table: it
shows each trigger's peak probability during that recording vs. its
threshold, plus the top-10 classes the model saw overall (useful for
figuring out what's actually being detected when a trigger doesn't fire).

Also useful for measuring raw inference speed on your machine:

```bash
python3 main.py --benchmark
```

## Tuning

Everything lives in `config.yaml`:

- `threshold` per trigger (0.0–1.0) — lower = more sensitive (mutes more
  eagerly, more false positives), higher = less sensitive. Tune with real
  numbers from `debug_listen.py`, not guesswork.
- `delay_s` — playback is intentionally delayed (default 1.3s) so a mute
  decision can be applied *before* you hear the sound, rather than always
  leaking the first fraction of a second of every trigger. Lower this and
  you'll hear more of each trigger's onset; raise it and the whole feed feels
  laggier.
- `attack_ms` / `release_ms` — fade speed in/out of a mute, avoids audible
  clicks.
- `hold_ms` — keeps muting for a bit after the last detection, to bridge
  brief gaps mid-sound (e.g. between chews) instead of chattering on/off.
- `block_ms` / `window_s` — how often inference runs and how much context it
  sees. Longer window = more context but more latency and CPU per call; check
  `--benchmark` stays comfortably under `block_ms` after changing these.

## Troubleshooting

**A trigger never fires, even when you're clearly making the sound.**
Run `debug_listen.py` while making that sound and check its `max_prob`
against its threshold:
- `max_prob` is a real number close to (but under) the threshold → lower the
  threshold for that trigger.
- `max_prob` is essentially `0.000` and nothing sound-related shows up in the
  "Top 10" list → the audio quality/level reaching the model is bad (see
  next two issues), or the class mapping doesn't match how your specific
  version of that sound actually sounds.

**Peak level reads near 0.0 in `debug_listen.py`, even though you made
noise.** The mic isn't picking anything up. Most common cause on macOS:
**device index drift**. CoreAudio reshuffles device indices whenever a
Bluetooth device connects or disconnects — if your config or a script has a
numeric device index hardcoded, it can silently start pointing at a
different (dead/virtual) device. Always use name substrings in
`config.yaml`, and re-run `list_devices.py` if in doubt; every script here
resolves devices by name fresh on every start for exactly this reason.

**Peak level is above 1.0.** Your input is clipping — the mic gain is too
high. This isn't just quieter/louder, it changes the *shape* of the
waveform (hard clipping), which can make the model misclassify a sound
entirely (in testing, a badly clipped clap was classified as a gunshot).
Lower the input volume in System Settings → Sound → Input, then re-check
peak level. Some Macs also apply automatic gain control that boosts a quiet
room a lot, then clips hard on a sudden loud sound — if peak level jumps
wildly between quiet and loud test clips, that's likely what's happening;
leave some headroom rather than tuning gain for the quiet case only.

**Detected classes are dominated by "Speech"/"Chatter"/"Crowd"/"Music" no
matter what you test.** There's probably background noise (conversation,
music, a call) loud enough to mask the trigger sound in the mix. Test
somewhere quieter.

**The web dashboard's Start button shows an error.** Almost always means the
configured output (or input) device isn't currently connected — the error
message names which one and lists what's currently available. Reconnect the
device and try again.

## Architecture

```
suppressor.py   shared pipeline: device resolution, TriggerDetector (PANNs
                wrapper), GainSmoother, and the Suppressor class that owns
                the live start()/stop()-able mic-in -> detect -> mute ->
                play-out loop.
main.py         thin CLI wrapper around Suppressor.
app.py          local Flask dashboard (same Suppressor, controlled over
                a tiny JSON API + polling UI instead of a terminal).
debug_listen.py records N seconds and prints per-trigger model confidence,
                for tuning without guesswork.
list_devices.py prints current CoreAudio device names/indices.
config.yaml     trigger->class mappings, thresholds, audio settings.
```

The model only sees the current trigger classes/thresholds each time it's
asked to load; the web dashboard's threshold sliders update the same
in-memory config the CLI uses, so both stay in sync with whatever's in
`config.yaml` at startup.

## Known limitations

- **Not a training pipeline.** This repurposes a general-purpose pretrained
  audio tagger (PANNs) rather than training anything custom — no dataset
  curation or GPU training involved. Good enough for the 6 configured
  triggers; a genuinely novel sound with no reasonable AudioSet class match
  won't be detectable this way.
- **~1.3s inherent latency.** The model needs to hear a sound before it can
  classify it, so there's a fundamental delay between a sound happening and
  a mute decision being possible. This is mitigated (not eliminated) by
  delaying playback so the decision lands before you hear it.
- **`tapping_on_phone` and `heavy_breathing` are proxy mappings**, not exact
  matches — may need more threshold tuning than the others, and may
  occasionally fire on similar-sounding but unrelated sounds (e.g. any
  generic tap/click for the former).
- **CPU inference only** — tested on Apple Silicon (M2), ~40ms per inference
  call against a 100ms budget, comfortable headroom. No GPU/MPS required,
  though `--device mps` is available if you want to experiment.
