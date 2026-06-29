#!/bin/bash
# Daily tournament update: refresh data, snapshot odds, re-forecast, rebuild site.
# Uses .venv/bin/python locally; CI overrides with PYTHON=python.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"

# Resilient fetch: martj42/eloratings are externally hosted and intermittently
# flaky, and under `set -e` a bare `curl -sL` made a transient blip fatal (the
# dominant cause of red runs at network-contended UTC hours). `curl -sL` also
# does NOT fail on an HTTP 5xx — it writes the error page straight over the
# committed snapshot, which a downstream parser then chokes on.
#   -f                    -> non-zero on HTTP >=400, so 5xx is caught not saved
#   --retry/--retry-*     -> ride out transient network/5xx blips
#   --*-timeout           -> never hang a runner forever
#   tmp file + [ -s ]     -> only replace the snapshot with a non-empty download
# On ultimate failure we keep the existing committed snapshot and warn LOUDLY
# (GitHub annotation) rather than abort — running one cycle on yesterday's data
# matches the repo's resilience intent, but must stay visible (CLAUDE.md: a green
# pipeline must never silently hide stale data). If no cached file exists at all,
# that IS fatal.
fetch() {
  local url="$1" dest="$2" tmp
  tmp="$(mktemp)"
  if curl -fsSL --retry 5 --retry-all-errors --retry-delay 3 \
          --connect-timeout 15 --max-time 180 -o "$tmp" "$url" && [ -s "$tmp" ]; then
    mv "$tmp" "$dest"
    echo "  fetched $(basename "$dest") ($(wc -c <"$dest" | tr -d ' ') bytes)"
  else
    rm -f "$tmp"
    if [ -s "$dest" ]; then
      echo "::warning::fetch failed for $url — keeping existing $(basename "$dest")" >&2
    else
      echo "::error::fetch failed for $url and no cached $dest exists" >&2
      return 1
    fi
  fi
}

echo "== refresh results & Elo =="
for f in results.csv shootouts.csv goalscorers.csv; do
  fetch "https://raw.githubusercontent.com/martj42/international_results/master/$f" "data/raw/$f"
done
fetch "https://eloratings.net/World.tsv" "data/raw/elo_world.tsv"

echo "== overlay live results (football-data.org; martj42 is the fallback) =="
# fetch_results.py already returns 0 for every tolerable case (missing key,
# network/auth error) and falls back to the martj42 snapshot. The ONLY way it
# exits non-zero is the deliberate FAIL-LOUDLY on an unmapped WC team name — a
# code bug that silently freezes results. Let that halt the run (set -e) instead
# of swallowing it with `|| echo continuing`, which is how a one-line name gap
# froze the published results for a day behind a green pipeline.
"$PY" scripts/fetch_results.py

echo "== snapshot odds =="
"$PY" scripts/snapshot_odds.py || echo "(odds snapshot failed; continuing)"

echo "== forecast (conditional on played results) =="
"$PY" scripts/run_forecast.py --sims 100000

echo "== golden boot race =="
"$PY" scripts/run_golden_boot.py

echo "== consensus =="
"$PY" scripts/run_consensus.py

echo "== append probability history + match predictions =="
"$PY" scripts/append_history.py

echo "== odds-movement tracker =="
"$PY" scripts/analyze_odds_movement.py

echo "== site =="
"$PY" scripts/build_site.py
echo "Done. Commit and push to publish."
