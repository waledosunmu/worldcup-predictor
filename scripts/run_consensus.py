"""Bookmaker-consensus forecast from the latest odds snapshots in data/odds/.

Produces output/consensus.json: outright champion probabilities and per-match
W/D/L consensus for all upcoming fixtures, plus a model-vs-market comparison
and a 50/50 log-space ensemble (unvalidated blend — no historical odds exist
to backtest it; both components are shown everywhere).

Usage: python scripts/run_consensus.py
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpred import history
from wcpred.names import canon
from wcpred.odds import event_h2h_consensus, outright_consensus
from wcpred.odds_movement import build_record


def latest(pattern: str, required: bool = True) -> tuple:
    files = sorted(glob.glob(str(ROOT / "data/odds" / pattern)))
    if not files:
        if required:
            raise SystemExit(f"no snapshot matching {pattern}; run scripts/snapshot_odds.py")
        return None, None
    return json.load(open(files[-1])), Path(files[-1]).name


def main():
    forecast = json.load(open(ROOT / "output/forecast_opening.json"))
    teams48 = set(forecast["ratings"])

    winner_snap, winner_file = latest("soccer_fifa_world_cup_winner_*.json")
    # Match-odds blobs are gitignored (compact-derived-only) and only on disk right
    # after a snapshot. When snapshot_odds skips (kickoff-aware) there's no fresh one
    # on a CI checkout — reuse the per-match consensus already in consensus.json rather
    # than failing; the outright below is still recomputed from the committed winner snap.
    match_snap, match_file = latest("soccer_fifa_world_cup_2026*.json", required=False)

    outright, n_books = outright_consensus(winner_snap["data"][0], canon=canon,
                                           universe=teams48)

    matches = []
    if match_snap is not None:
        for ev in match_snap["data"]:
            c = event_h2h_consensus(ev)
            if c is None:
                continue
            c["home"], c["away"] = canon(c["home"]), canon(c["away"])
            c["commence"] = ev["commence_time"]
            if c["home"] not in teams48 or c["away"] not in teams48:
                print(f"WARNING unmatched match teams: {c['home']} vs {c['away']}")
                continue
            # attach model probs + ensemble for the same fixture
            model = next((f for f in forecast["group_fixtures"]
                          if f["home"] == c["home"] and f["away"] == c["away"]), None)
            if model:
                m = np.array([model["p_home"], model["p_draw"], model["p_away"]])
                k = np.array([c["p_home"], c["p_draw"], c["p_away"]])
                ens = np.exp((np.log(np.clip(m, 1e-9, 1)) + np.log(np.clip(k, 1e-9, 1))) / 2)
                ens /= ens.sum()
                c["model"] = {"p_home": model["p_home"], "p_draw": model["p_draw"],
                              "p_away": model["p_away"]}
                c["ensemble"] = {"p_home": float(ens[0]), "p_draw": float(ens[1]),
                                 "p_away": float(ens[2])}
            matches.append(c)
    else:
        prev = ROOT / "output/consensus.json"
        if prev.exists():
            matches = json.load(open(prev)).get("matches", [])
        match_file = f"(reused {len(matches)} markets from previous consensus.json)"
        print(f"no fresh match-odds snapshot — {match_file}")

    model_champ = {t: forecast["advancement"][t]["champion"] for t in teams48}
    comparison = sorted(
        ({"team": t, "market": outright.get(t, 0.0), "model": model_champ[t],
          "diff": model_champ[t] - outright.get(t, 0.0)} for t in teams48),
        key=lambda r: -r["market"])

    out = {"captured": winner_snap["captured"], "sources": [winner_file, match_file],
           "n_outright_books": n_books, "outright_consensus": outright,
           "champion_comparison": comparison, "matches": matches}
    with open(ROOT / "output/consensus.json", "w") as f:
        json.dump(out, f, indent=1)

    print(f"Outright consensus from {n_books} full-field books; "
          f"{len(matches)} match markets")
    print(f"{'team':15s} {'market':>8s} {'model':>8s}")
    for r in comparison[:10]:
        print(f"{r['team']:15s} {r['market']:8.1%} {r['model']:8.1%}")
    big = sorted(comparison, key=lambda r: -abs(r["diff"]))[:5]
    print("largest model-market divergences:",
          ", ".join(f"{r['team']} {r['diff']:+.1%}" for r in big))
    print("Wrote output/consensus.json")

    # compact odds-movement time-series (one row per fresh capture; idempotent).
    # Skip when matches were reused (no fresh snapshot) — nothing new to record.
    ts_path = ROOT / "output/odds_timeseries.jsonl"
    series = history.load_jsonl(ts_path)
    if match_snap is not None \
            and not any(r["captured"] == winner_snap["captured"] for r in series):
        series.append(build_record(winner_snap["captured"], n_books, outright, matches))
        history.write_jsonl(ts_path, sorted(series, key=lambda r: r["captured"]))
        print(f"odds_timeseries: appended capture {winner_snap['captured']} "
              f"({len(series)} total)")


if __name__ == "__main__":
    main()
