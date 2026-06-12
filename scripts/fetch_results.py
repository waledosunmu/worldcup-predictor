"""Overlay live 2026 World Cup results onto the committed snapshot.

The forecast (scripts/run_forecast.py) already locks any WC2026 row that carries
a real score; the committed martj42 snapshot ships those rows as NA stubs and
martj42 fills them with a lag. This script fetches finished matches from
football-data.org (a faster live feed) and writes their scores onto the matching
NA rows of data/raw/results.csv, recording penalty-shootout winners in
data/raw/shootouts.csv. It never overwrites an existing result, so when martj42
itself updates (the fallback path) nothing here clobbers it.

Requires a free football-data.org token in FOOTBALL_DATA_API_KEY (env or .env,
same loader as snapshot_odds.py). Without a key it prints a notice and exits 0 so
the daily pipeline keeps working — and the martj42 re-download in update_all.sh
remains the fallback source of truth.

Usage: FOOTBALL_DATA_API_KEY=... python scripts/fetch_results.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpred import live
from wcpred.simulate import load_format


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and report, but do not write the CSVs")
    args = ap.parse_args()

    live.load_env(ROOT)
    import os
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        print("FOOTBALL_DATA_API_KEY not set — skipping live overlay "
              "(martj42 snapshot remains the source of truth). "
              "Free key: https://www.football-data.org/client/register")
        return

    fmt = load_format(ROOT / "data/format_2026.json")
    valid = live.squad_names(fmt)

    try:
        matches = live.fetch_football_data(key)
    except Exception as e:  # network / auth / tier error — never break the pipeline
        print(f"live fetch failed ({e}) — martj42 snapshot remains the source of truth")
        return
    print(f"fetched {len(matches)} finished World Cup matches from football-data.org")

    results = pd.read_csv(ROOT / "data/raw/results.csv")
    shootouts = pd.read_csv(ROOT / "data/raw/shootouts.csv")
    results, shootouts, summary = live.merge_results(results, shootouts, matches, valid)

    if summary["unmapped"]:
        # loud, non-silent failure: an unmapped WC team means a missed join,
        # which would silently reproduce the "0 played" bug. Fix LIVE_NAME_MAP.
        sys.exit(f"ERROR: unmapped team names from live feed: "
                 f"{sorted(set(summary['unmapped']))} — add them to "
                 f"wcpred.live.LIVE_NAME_MAP")

    print(f"  matched {summary['matched']} to WC fixtures; "
          f"filled {summary['filled']} NA scores; "
          f"{summary['already_had_score']} already had results; "
          f"{summary['no_fixture']} had no WC fixture; "
          f"added {summary['shootouts_added']} shootout records")

    if args.dry_run:
        print("  (dry run — CSVs not written)")
        return
    wrote = []
    if summary["filled"]:
        live.write_results_csv(results, ROOT / "data/raw/results.csv")
        wrote.append("results.csv")
    if summary["shootouts_added"]:
        shootouts.to_csv(ROOT / "data/raw/shootouts.csv", index=False)
        wrote.append("shootouts.csv")
    print(f"  wrote {', '.join('data/raw/' + w for w in wrote)}"
          if wrote else "  nothing new to write")


if __name__ == "__main__":
    main()
