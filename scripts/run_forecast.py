"""Conditional tournament forecast: lock in played 2026 results, simulate the rest.

Elo ratings update through the latest completed match (tournament games move
ratings, as they should); played group scores and knockout winners (shootouts
resolved via shootouts.csv) are fixed in every simulation; only the remaining
matches are sampled. With no played matches this reproduces the opening
forecast methodology exactly.

Usage: python scripts/run_forecast.py [--as-of YYYY-MM-DD] [--sims 100000]
       (--as-of defaults to today; matches dated before it count as played)
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpred import elo, goals
from wcpred.simulate import HOSTS, Simulator, load_format

FIT_WINDOW_YEARS = 28


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=dt.date.today().isoformat())
    ap.add_argument("--sims", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=26)
    args = ap.parse_args()
    as_of = args.as_of

    results = pd.read_csv(ROOT / "data/raw/results.csv")
    shootouts = pd.read_csv(ROOT / "data/raw/shootouts.csv")
    fmt = load_format(ROOT / "data/format_2026.json")
    teams48 = [t for g in sorted(fmt["groups"]) for t in fmt["groups"][g]]
    team_group = {t: g for g, ts in fmt["groups"].items() for t in ts}

    # ---- played 2026 matches ----
    wc26 = results[(results.tournament == "FIFA World Cup")
                   & (results.date >= "2026-06-01") & (results.date < as_of)]
    wc26 = wc26.dropna(subset=["home_score", "away_score"])
    so_winner = {frozenset((r.home_team, r.away_team)): r.winner
                 for _, r in shootouts[shootouts.date >= "2026-06-01"].iterrows()}
    played_group, ko_winners = {}, {}
    for _, r in wc26.iterrows():
        same_group = team_group[r.home_team] == team_group[r.away_team]
        key = frozenset((r.home_team, r.away_team))
        if same_group and r.date <= fmt["dates"]["group_stage"][-10:]:
            played_group[key] = (r.home_team, int(r.home_score), int(r.away_score))
        else:
            if r.home_score != r.away_score:
                ko_winners[key] = r.home_team if r.home_score > r.away_score \
                    else r.away_team
            elif key in so_winner:
                ko_winners[key] = so_winner[key]
            else:
                print(f"WARNING: drawn KO match without shootout record: "
                      f"{r.home_team} vs {r.away_team} — left unfixed")
    print(f"as of {as_of}: {len(played_group)} group results, "
          f"{len(ko_winners)} knockout results locked in")

    # ---- point-in-time ratings and goals model ----
    hist = elo.rating_history_frame(results[results.date < as_of])
    ratings = elo.compute_ratings(results, as_of=as_of)
    fit_df = hist[(hist.date >= f"{int(as_of[:4]) - FIT_WINDOW_YEARS}-01-01")
                  & (hist.tournament != "Friendly")]
    params = goals.fit(fit_df.dr.to_numpy(),
                       fit_df.home_score.to_numpy(), fit_df.away_score.to_numpy())
    print(f"goals model: a={params['a']:.4f} b={params['b']:.4f} n={params['n']}")

    # ---- simulate ----
    sim = Simulator(fmt, ratings, params, seed=args.seed,
                    played_group=played_group, ko_winners=ko_winners)
    out = sim.run(args.sims)

    fixtures = []
    for g, home, away, lh, la in sim.group_fixtures:
        rec = played_group.get(frozenset((home, away)))
        fx = {"group": g, "home": home, "away": away,
              "elo_home": round(ratings[home]), "elo_away": round(ratings[away])}
        if rec is not None:
            first, gf, ga = rec
            sh, sa = (gf, ga) if first == home else (ga, gf)
            fx.update(played=True, score_home=sh, score_away=sa)
        else:
            dr = ratings[home] - ratings[away] + (100.0 if home in HOSTS else 0.0)
            pw, pd_, pl = goals.outcome_probs(dr, params)
            fx.update(played=False, xg_home=round(lh, 2), xg_away=round(la, 2),
                      p_home=round(pw, 4), p_draw=round(pd_, 4), p_away=round(pl, 4))
        fixtures.append(fx)

    payload = {
        "generated": as_of, "as_of": as_of, "model": "elo-poisson-v0-conditional",
        "n_sims": out["n_sims"], "params": params,
        "n_played_group": len(played_group), "n_played_ko": len(ko_winners),
        "ratings": {t: round(ratings[t], 1) for t in teams48},
        "advancement": out["probs"], "group_fixtures": fixtures,
    }
    with open(ROOT / "output/forecast_latest.json", "w") as f:
        json.dump(payload, f, indent=1)
    print("Wrote output/forecast_latest.json — top 8:")
    rows = sorted(out["probs"].items(), key=lambda kv: -kv[1]["champion"])
    for i, (t, p) in enumerate(rows[:8], 1):
        print(f"  {i}. {t:15s} champion {p['champion']:.1%}")


if __name__ == "__main__":
    main()
