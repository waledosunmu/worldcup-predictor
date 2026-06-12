#!/bin/bash
# Daily tournament update: refresh data, snapshot odds, re-forecast, rebuild site.
# Uses .venv/bin/python locally; CI overrides with PYTHON=python.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"

echo "== refresh results & Elo =="
for f in results.csv shootouts.csv; do
  curl -sL -o "data/raw/$f" "https://raw.githubusercontent.com/martj42/international_results/master/$f"
done
curl -sL -o data/raw/elo_world.tsv "https://eloratings.net/World.tsv"

echo "== overlay live results (football-data.org; martj42 is the fallback) =="
"$PY" scripts/fetch_results.py || echo "(live results overlay failed; continuing)"

echo "== snapshot odds =="
"$PY" scripts/snapshot_odds.py || echo "(odds snapshot failed; continuing)"

echo "== forecast (conditional on played results) =="
"$PY" scripts/run_forecast.py --sims 100000

echo "== consensus =="
"$PY" scripts/run_consensus.py

echo "== append probability history + match predictions =="
"$PY" scripts/append_history.py

echo "== odds-movement tracker =="
"$PY" scripts/analyze_odds_movement.py

echo "== site =="
"$PY" scripts/build_site.py
echo "Done. Commit and push to publish."
