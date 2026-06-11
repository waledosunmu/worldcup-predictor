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

    sports, meta = get("/sports", all="true")
    wc_keys = [s["key"] for s in sports
               if "world cup" in s.get("title", "").lower()
               or "world_cup" in s["key"]]
    print(f"world-cup sport keys: {wc_keys} ({meta})")

    for key in wc_keys:
        markets = "outrights" if key.endswith("winner") else "h2h"
        try:
            data, meta = get(f"/sports/{key}/odds", regions="eu,us,uk",
                             markets=markets, oddsFormat="decimal")
        except Exception as e:  # key without active markets
            print(f"  {key}: skipped ({e})")
            continue
        path = outdir / f"{key}_{stamp}.json"
        path.write_text(json.dumps(
            {"captured": stamp, "sport_key": key, "markets": markets, "data": data},
            indent=1))
        print(f"  {key}: {len(data)} events -> {path.name} ({meta})")


if __name__ == "__main__":
    main()
