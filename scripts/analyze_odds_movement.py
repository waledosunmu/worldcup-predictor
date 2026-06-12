"""Report how much the betting market moves in the hours before kickoff.

Reads output/odds_timeseries.jsonl (compact per-capture consensus, written by
run_consensus.py) and writes output/odds_movement.{json,md} — drift metrics plus a
plain-English verdict on whether the near-kickoff snapshot cadence is buying signal.

--backfill first reconstructs the time-series from any raw data/odds snapshot pairs
still on disk (the ~1.4 MB blobs are gitignored but linger locally), so the report has
history even before the higher cadence has run.

Usage: python scripts/analyze_odds_movement.py [--backfill]
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpred import history, odds_movement
from wcpred.names import canon
from wcpred.odds import snapshot_to_consensus

TS_PATH = ROOT / "output/odds_timeseries.jsonl"
_STAMP = re.compile(r"_(\d{4}-\d{2}-\d{2}T\d{6}Z)\.json$")


def backfill_from_snapshots() -> list[dict]:
    """Reconstruct compact time-series rows from raw on-disk snapshot pairs."""
    opening = json.load(open(ROOT / "output/forecast_opening.json"))
    teams48 = set(opening["ratings"])
    series = history.load_jsonl(TS_PATH)
    seen = {r["captured"] for r in series}

    def by_stamp(pattern: str) -> dict:
        out = {}
        for f in glob.glob(str(ROOT / "data/odds" / pattern)):
            m = _STAMP.search(f)
            if m:
                out[m.group(1)] = f
        return out

    winners = by_stamp("soccer_fifa_world_cup_winner_*.json")
    matches = by_stamp("soccer_fifa_world_cup_2026*.json")
    added = 0
    for stamp in sorted(set(winners) & set(matches)):
        if stamp in seen:
            continue
        c = snapshot_to_consensus(json.load(open(winners[stamp])),
                                  json.load(open(matches[stamp])),
                                  canon=canon, universe=teams48)
        series.append(odds_movement.build_record(stamp, c["n_outright_books"],
                                                  c["outright_consensus"], c["matches"]))
        added += 1
    series.sort(key=lambda r: r["captured"])
    history.write_jsonl(TS_PATH, series)
    print(f"backfill: added {added} capture(s) from raw snapshots ({len(series)} total)")
    return series


def render_md(a: dict) -> str:
    lines = ["# Odds-movement tracker — pre-kickoff market drift", "",
             f"_{a['n_records']} capture(s); {a['n_matches_tracked']} match(es) with "
             f"≥2 pre-kickoff captures._", "", f"**{a['verdict']}**", ""]
    if a["n_matches_tracked"]:
        t, f = a["total_drift_pp"], a["final_window_move_pp"]
        fmean = "n/a" if f["mean"] is None else f["mean"]
        fmax = "n/a" if f["max"] is None else f["max"]
        lines += [
            "| Metric | Mean | Median | Max |",
            "|---|---:|---:|---:|",
            f"| Total pre-kickoff drift (pp) | {t['mean']} | {t['median']} | {t['max']} |",
            f"| Final-{odds_movement.FINAL_WINDOW_H:g}h move (pp), {f['n_matches']} match(es)"
            f" | {fmean} | — | {fmax} |",
            "",
            f"Mean captures per match: {a['mean_captures_per_match']}. "
            f"Max outright champion-prob swing across captures: {a['outright_max_swing_pp']}pp.",
            "", "## Biggest movers (favourite probability)", "",
            "| Match | Captures | Fav | Open | Close | Drift (pp) | Final-window (pp) |",
            "|---|--:|--|--:|--:|--:|--:|"]
        for r in a["per_match"][:12]:
            fw = "—" if r["final_window_move_pp"] is None else r["final_window_move_pp"]
            lines.append(
                f"| {r['home']} v {r['away']} | {r['n_captures']} | {r['favourite']} | "
                f"{r['open_prob']:.0%} | {r['close_prob']:.0%} | {r['total_drift_pp']} | "
                f"{fw} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="reconstruct the series from raw on-disk snapshots first")
    args = ap.parse_args()

    series = backfill_from_snapshots() if args.backfill else history.load_jsonl(TS_PATH)
    analysis = odds_movement.analyze(series)

    (ROOT / "output/odds_movement.json").write_text(json.dumps(analysis, indent=1))
    (ROOT / "output/odds_movement.md").write_text(render_md(analysis))
    print(analysis["verdict"])
    print("Wrote output/odds_movement.{json,md}")


if __name__ == "__main__":
    main()
