"""
Local web dashboard for the misophonia trigger suppressor.

Run:
    python3 app.py

Then open http://127.0.0.1:5757 in your browser. The audio pipeline itself
still runs entirely in this Python process (only it can talk to your
CoreAudio devices) -- the page just gives you Start/Stop, live status, and
threshold sliders instead of editing config.yaml and watching a terminal.
"""
import sys

from flask import Flask, jsonify, request

from suppressor import Suppressor, load_config

app = Flask(__name__)
cfg = load_config("config.yaml")

print("Loading model (one-time, happens once at server startup)...", file=sys.stderr)
sup = Suppressor(cfg, device="cpu")
print("Model loaded. Dashboard ready.", file=sys.stderr)


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Misophonia Suppressor</title>
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 640px; margin: 40px auto; padding: 0 20px;
    background: Canvas; color: CanvasText;
  }
  h1 { font-size: 1.4rem; margin-bottom: 4px; }
  .sub { color: #888; font-size: 0.9rem; margin-bottom: 24px; }
  .card {
    border: 1px solid #8883; border-radius: 12px; padding: 20px;
    margin-bottom: 20px;
  }
  .row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .pill {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: 0.85rem; font-weight: 600;
  }
  .pill.running { background: #1a7f3722; color: #1a7f37; }
  .pill.stopped { background: #8883; color: #888; }
  .pill.muted { background: #cf222e22; color: #cf222e; }
  button {
    font-size: 0.95rem; padding: 10px 22px; border-radius: 8px; border: none;
    cursor: pointer; font-weight: 600;
  }
  button.start { background: #1a7f37; color: white; }
  button.stop { background: #cf222e; color: white; }
  button:disabled { opacity: 0.5; cursor: default; }
  .devices { font-size: 0.85rem; color: #888; margin-top: 10px; }
  .error {
    background: #cf222e18; color: #cf222e; border-radius: 8px; padding: 10px 14px;
    font-size: 0.9rem; margin-bottom: 16px; display: none;
  }
  table { width: 100%; border-collapse: collapse; }
  td { padding: 8px 0; font-size: 0.9rem; }
  td.name { font-weight: 600; }
  td.classes { color: #888; font-size: 0.78rem; }
  input[type=range] { width: 100%; }
  .thresh-val { font-variant-numeric: tabular-nums; font-size: 0.85rem; color: #888; }
  ul#events { list-style: none; padding: 0; margin: 0; font-size: 0.85rem; }
  ul#events li { padding: 6px 0; border-bottom: 1px solid #8882; display: flex; justify-content: space-between; }
  ul#events li:last-child { border-bottom: none; }
  #empty-events { color: #888; font-size: 0.85rem; }
</style>
</head>
<body>
  <h1>🔇 Misophonia Trigger Suppressor</h1>
  <div class="sub">Live mic → mute triggers → headphones, running locally.</div>

  <div id="error" class="error"></div>

  <div class="card">
    <div class="row">
      <span id="status-pill" class="pill stopped">Stopped</span>
      <button id="toggle-btn" class="start" onclick="toggle()">Start</button>
    </div>
    <div class="devices" id="devices"></div>
  </div>

  <div class="card">
    <div class="row" style="margin-bottom: 10px;">
      <strong>Recent detections</strong>
    </div>
    <ul id="events"></ul>
    <div id="empty-events">Nothing detected yet.</div>
  </div>

  <div class="card">
    <div class="row" style="margin-bottom: 14px;">
      <strong>Trigger sensitivity</strong>
    </div>
    <table id="thresholds"></table>
    <div class="devices">Lower = more sensitive (mutes more easily). Changes apply immediately.</div>
  </div>

<script>
let thresholdsBuilt = false;

function fmtTime(t) {
  if (!t) return '';
  return new Date(t * 1000).toLocaleTimeString();
}

async function refresh() {
  try {
    const res = await fetch('/api/status');
    const st = await res.json();
    document.getElementById('error').style.display = 'none';

    const pill = document.getElementById('status-pill');
    const btn = document.getElementById('toggle-btn');
    if (st.running) {
      pill.textContent = 'Running';
      pill.className = 'pill running';
      btn.textContent = 'Stop';
      btn.className = 'stop';
    } else {
      pill.textContent = 'Stopped';
      pill.className = 'pill stopped';
      btn.textContent = 'Start';
      btn.className = 'start';
    }
    btn.disabled = false;

    document.getElementById('devices').textContent = st.running
      ? `Input: ${st.input_device}  →  Output: ${st.output_device}`
      : 'Not running';

    const list = document.getElementById('events');
    const empty = document.getElementById('empty-events');
    list.innerHTML = '';
    if (st.recent_events.length === 0) {
      empty.style.display = 'block';
    } else {
      empty.style.display = 'none';
      for (const e of st.recent_events) {
        const li = document.createElement('li');
        li.innerHTML = `<span>${e.trigger}</span><span>${fmtTime(e.time)}</span>`;
        list.appendChild(li);
      }
    }

    if (!thresholdsBuilt) {
      const table = document.getElementById('thresholds');
      for (const [name, val] of Object.entries(st.thresholds)) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="name">${name}<br><span class="classes">${st.classes[name].join(', ')}</span></td>
          <td style="width: 45%;">
            <input type="range" min="0.02" max="0.9" step="0.01" value="${val}"
                   oninput="this.nextElementSibling.textContent = this.value"
                   onchange="setThreshold('${name}', this.value)">
            <div class="thresh-val">${val}</div>
          </td>`;
        table.appendChild(tr);
      }
      thresholdsBuilt = true;
    }
  } catch (e) {
    // server not reachable yet (e.g. still loading model) -- keep polling
  }
}

async function toggle() {
  const btn = document.getElementById('toggle-btn');
  const isRunning = btn.textContent === 'Stop';
  btn.disabled = true;
  const res = await fetch(isRunning ? '/api/stop' : '/api/start', { method: 'POST' });
  const data = await res.json();
  if (!data.ok) {
    const err = document.getElementById('error');
    err.textContent = data.error;
    err.style.display = 'block';
  }
  refresh();
}

async function setThreshold(name, value) {
  await fetch('/api/threshold', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, value: parseFloat(value) }),
  });
}

refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return PAGE


@app.route("/api/status")
def api_status():
    return jsonify({"ok": True, **sup.status()})


@app.route("/api/start", methods=["POST"])
def api_start():
    try:
        sup.start()
        return jsonify({"ok": True, **sup.status()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), **sup.status()}), 200


@app.route("/api/stop", methods=["POST"])
def api_stop():
    sup.stop()
    return jsonify({"ok": True, **sup.status()})


@app.route("/api/threshold", methods=["POST"])
def api_threshold():
    data = request.get_json()
    try:
        sup.set_threshold(data["name"], data["value"])
        return jsonify({"ok": True, **sup.status()})
    except KeyError:
        return jsonify({"ok": False, "error": f"Unknown trigger '{data.get('name')}'"}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5757, debug=False)
