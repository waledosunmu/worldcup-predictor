#!/bin/bash
# Daily tournament update: refresh data, snapshot odds, re-forecast, rebuild site.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== refresh results & Elo =="
for f in results.csv shootouts.csv; do
  curl -sL -o "data/raw/$f" "https://raw.githubusercontent.com/martj42/international_results/master/$f"
done
curl -sL -o data/raw/elo_world.tsv "https://eloratings.net/World.tsv"

echo "== snapshot odds =="
.venv/bin/python scripts/snapshot_odds.py || echo "(odds snapshot failed; continuing)"

echo "== forecast =="
.venv/bin/python scripts/run_opening_forecast.py --sims 100000

echo "== consensus =="
.venv/bin/python scripts/run_consensus.py

echo "== site =="
.venv/bin/python scripts/build_site.py
echo "Done. Commit and push to publish."
