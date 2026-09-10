#!/bin/zsh
# Double-click this file in Finder to start the suppressor and open the
# dashboard in your browser automatically. Leave this window open while
# it runs; close it (or Ctrl+C) to stop the server.

cd "$(dirname "$0")"

if [ ! -d venv ]; then
  echo "venv/ not found -- run setup first (see README.md: python3 -m venv venv && pip install -r requirements.txt)."
  read "?Press Enter to close..."
  exit 1
fi

source venv/bin/activate

if curl -s -o /dev/null -m 1 "http://127.0.0.1:5757/api/status" 2>/dev/null; then
  echo "Already running -- opening the dashboard."
  open "http://127.0.0.1:5757"
  exit 0
fi

# Open the browser automatically once the server is actually ready, instead
# of guessing a fixed delay (model load time varies).
(
  for _ in $(seq 1 60); do
    if curl -s -o /dev/null -m 1 "http://127.0.0.1:5757/api/status" 2>/dev/null; then
      open "http://127.0.0.1:5757"
      break
    fi
    sleep 1
  done
) &

echo "Starting Misophonia Suppressor... the dashboard will open automatically."
echo "Leave this window open. Close it or press Ctrl+C to stop."
echo ""

python3 app.py
