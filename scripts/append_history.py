"""Append one probability-history point and refresh match predictions.

Runs after run_consensus.py in update_all.sh. Reads output/forecast_latest.json (the
live model) and output/consensus.json (the market), then updates:
  - output/probability_history.jsonl   (per as_of; trend graph + movers)
  - output/match_predictions.jsonl     (per fixture; live track-record)

--backfill additionally seeds the opening-day point and the opening match predictions
from output/forecast_opening.json + the earliest data/odds/*_winner_*.json and match
snapshots, reconstructing the market consensus as of those snapshots. Idempotent — safe
to re-run.

Usage: python scripts/append_history.py [--backfill]
"""
import argparse
import datetime as dt
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpred import history
from wcpred.names import canon
from wcpred.odds import event_h2h_consensus, outright_consensus

HIST_PATH = ROOT / "output/probability_history.jsonl"
PRED_PATH = ROOT / "output/match_predictions.jsonl"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def reconstruct_consensus(winner_file: Path, match_file: Path, teams48: set) -> dict:
    """Build a consensus-shaped dict from raw odds snapshots (for backfill)."""
    winner_snap = json.load(open(winner_file))
    match_snap = json.load(open(match_file))
    outright, n_books = outright_consensus(winner_snap["data"][0], canon=canon,
                                           universe=teams48)
    matches = []
    for ev in match_snap["data"]:
        c = event_h2h_consensus(ev)
        if c is None:
            continue
        c["home"], c["away"] = canon(c["home"]), canon(c["away"])
        c["commence"] = ev["commence_time"]
        if c["home"] in teams48 and c["away"] in teams48:
            matches.append(c)
    return {"captured": winner_snap["captured"], "n_outright_books": n_books,
            "outright_consensus": outright, "matches": matches}


def backfill_opening(hist_records: list[dict], pred_records: list[dict]):
    """Seed the opening-day history point and opening match predictions."""
    opening = json.load(open(ROOT / "output/forecast_opening.json"))
    teams48 = set(opening["ratings"])
    opening_latest = {**opening,
                      "as_of": opening.get("as_of", opening["generated"]),
                      "n_played_group": 0, "n_played_ko": 0}

    def earliest(pattern: str):
        files = sorted(glob.glob(str(ROOT / "data/odds" / pattern)))
        return Path(files[0]) if files else None

    winner_f = earliest("soccer_fifa_world_cup_winner_*.json")
    match_f = earliest("soccer_fifa_world_cup_2026*.json")
    if winner_f is None or match_f is None:
        print("backfill: no opening odds snapshots found — seeding model point only")
        consensus = {"captured": opening["generated"], "n_outright_books": 0,
                     "outright_consensus": {}, "matches": []}
    else:
        consensus = reconstruct_consensus(winner_f, match_f, teams48)
        print(f"backfill: reconstructed opening market from {winner_f.name} / "
              f"{match_f.name} ({consensus['n_outright_books']} books, "
              f"{len(consensus['matches'])} matches)")

    rec = history.build_history_record(opening_latest, consensus, consensus["captured"])
    hist_records = history.upsert_history(hist_records, rec)
    preds = history.build_match_predictions(opening_latest, consensus, consensus["captured"])
    pred_records = history.merge_match_predictions(pred_records, preds, [])
    return hist_records, pred_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="seed opening-day point + predictions from forecast_opening.json")
    args = ap.parse_args()

    hist_records = history.load_jsonl(HIST_PATH)
    pred_records = history.load_jsonl(PRED_PATH)

    if args.backfill:
        hist_records, pred_records = backfill_opening(hist_records, pred_records)

    latest = json.load(open(ROOT / "output/forecast_latest.json"))
    consensus = json.load(open(ROOT / "output/consensus.json"))
    now = _now()

    rec = history.build_history_record(latest, consensus, now)
    hist_records = history.upsert_history(hist_records, rec)

    new_preds = history.build_match_predictions(latest, consensus, now)
    played = [fx for fx in latest.get("group_fixtures", []) if fx.get("played")]
    pred_records = history.merge_match_predictions(pred_records, new_preds, played)

    history.write_jsonl(HIST_PATH, hist_records)
    history.write_jsonl(PRED_PATH, pred_records)

    scored = sum(1 for p in pred_records if p["played"] and p["model"])
    print(f"history: {len(hist_records)} point(s) (latest as_of {rec['as_of']}); "
          f"predictions: {len(pred_records)} fixture(s), {scored} played-with-prediction")


if __name__ == "__main__":
    main()
