"""Snapshot 2026 World Cup bookmaker odds (outrights + match h2h) to data/odds/.

Requires a free API key from https://the-odds-api.com (500 requests/month free)
in the ODDS_API_KEY environment variable. Sport keys are discovered dynamically
so renamed keys don't break the script.

Run daily during the tournament: pre-match odds disappear once games kick off.

Usage: ODDS_API_KEY=... python scripts/snapshot_odds.py
"""

import datetime as dt
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://api.the-odds-api.com/v4"

# Strictly the men's 2026 FIFA World Cup finals: match h2h + outright winner.
# The-odds-api reuses these keys each cycle, so they resolve to 2026 right now.
# A broad "world cup" match would also pull qualifiers, the Club World Cup, the
# Women's World Cup and non-soccer World Cups — each an extra paid /odds call.
TARGET_SPORT_KEYS = {"soccer_fifa_world_cup", "soccer_fifa_world_cup_winner"}

_env = ROOT / ".env"
if "ODDS_API_KEY" not in os.environ and _env.exists():
    for _line in _env.read_text().splitlines():
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def get(path: str, **params) -> tuple[object, dict]:
    params["apiKey"] = os.environ["ODDS_API_KEY"]
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as r:
        remaining = r.headers.get("x-requests-remaining")
        return json.load(r), {"requests_remaining": remaining}


def main():
    if "ODDS_API_KEY" not in os.environ:
        sys.exit("Set ODDS_API_KEY (free key: https://the-odds-api.com)")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    outdir = ROOT / "data/odds"
    outdir.mkdir(parents=True, exist_ok=True)

    sports, meta = get("/sports", all="true")  # free endpoint, no quota cost
    listed = {s["key"]: s for s in sports}
    wc_keys = [k for k in TARGET_SPORT_KEYS
               if k in listed and listed[k].get("active")]
    missing = TARGET_SPORT_KEYS - listed.keys()
    if missing:
        print(f"warning: target sport key(s) not listed by the API: {sorted(missing)}")
    inactive = [k for k in TARGET_SPORT_KEYS
                if k in listed and not listed[k].get("active")]
    if inactive:
        print(f"note: skipping inactive market(s) (no live odds): {sorted(inactive)}")
    print(f"target sport keys: {sorted(wc_keys)} ({meta})")

    for key in wc_keys:
        markets = "outrights" if key.endswith("winner") else "h2h"
        try:
            data, meta = get(f"/sports/{key}/odds", regions="eu,us,uk",
                             markets=markets, oddsFormat="decimal")
        except Exception as e:  # key without active markets
            print(f"  {key}: skipped ({e})")
            continue
        if not data:
            print(f"  {key}: no events, not saved")
            continue
        path = outdir / f"{key}_{stamp}.json"
        path.write_text(json.dumps(
            {"captured": stamp, "sport_key": key, "markets": markets, "data": data},
            indent=1))
        print(f"  {key}: {len(data)} events -> {path.name} ({meta})")


if __name__ == "__main__":
    main()
