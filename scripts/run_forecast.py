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

from wcpred import dixoncoles, goals, pointwise
from wcpred.simulate import HOSTS, Simulator, load_format


def _grid_wdl(grid):
    n = grid.shape[0]
    i = np.arange(n)
    hh, aa = np.meshgrid(i, i, indexing="ij")
    return (float(grid[hh > aa].sum()), float(np.trace(grid)), float(grid[hh < aa].sum()))


def _top_scores(grid, k: int = 3):
    """The k most-likely (home, away) scorelines from a joint grid, descending."""
    n = grid.shape[0]
    flat = grid.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    return [{"h": int(i // n), "a": int(i % n), "p": round(float(flat[i]), 4)} for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=dt.date.today().isoformat())
    ap.add_argument("--sims", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--model", choices=["hybrid", "v0"], default="hybrid",
                    help="hybrid (default, production): recent-form covariate + "
                         "Dixon-Coles; v0: Elo-only independent Poisson")
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
    # Full played-knockout records (teams, score, winner, shootout flag) so the
    # site can render actual KO results — ko_winners keeps only the winner, which
    # is all the simulation needs but not enough to show a scoreline.
    ko_results = []
    for _, r in wc26.iterrows():
        same_group = team_group[r.home_team] == team_group[r.away_team]
        key = frozenset((r.home_team, r.away_team))
        if same_group and r.date <= fmt["dates"]["group_stage"][-10:]:
            played_group[key] = (r.home_team, int(r.home_score), int(r.away_score))
        else:
            sh, sa = int(r.home_score), int(r.away_score)
            if r.home_score != r.away_score:
                winner = r.home_team if r.home_score > r.away_score else r.away_team
                ko_winners[key] = winner
                shootout = False
            elif key in so_winner:
                winner = so_winner[key]
                ko_winners[key] = winner
                shootout = True
            else:
                print(f"WARNING: drawn KO match without shootout record: "
                      f"{r.home_team} vs {r.away_team} — left unfixed")
                continue
            ko_results.append({"date": r.date, "home": r.home_team,
                               "away": r.away_team, "played": True,
                               "score_home": sh, "score_away": sa,
                               "winner": winner, "shootout": shootout})
    ko_results.sort(key=lambda x: (x["date"], x["home"]))
    print(f"as of {as_of}: {len(played_group)} group results, "
          f"{len(ko_winners)} knockout results locked in")

    # ---- point-in-time ratings and goals model ----
    # Shared with the knockout backfill via wcpred.pointwise so both stay one
    # leak-free methodology (see that module).
    bundle = pointwise.build_asof_model(results, as_of, args.model)
    ratings = bundle["ratings"]
    params = bundle["params"]
    team_form = bundle["team_form"]
    hybrid = bundle["hybrid"]
    model_name = bundle["model_name"]
    if hybrid:
        print(f"hybrid goals model: a={params['a']:.4f} b={params['b']:.4f} "
              f"form_beta={params['beta']['form_diff']:.4f} rho={params['rho']:.4f} "
              f"n={params['n']}")
    else:
        print(f"v0 goals model: a={params['a']:.4f} b={params['b']:.4f} n={params['n']}")

    # ---- simulate ----
    sim = Simulator(fmt, ratings, params, seed=args.seed,
                    played_group=played_group, ko_winners=ko_winners,
                    dixon_coles=hybrid, team_form=team_form)
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
            if hybrid:  # lh, la already carry the form covariate
                grid = dixoncoles.scoreline_grid(lh, la, params["rho"])
                pw, pd_, pl = _grid_wdl(grid)
            else:
                dr = ratings[home] - ratings[away] + (100.0 if home in HOSTS else 0.0)
                pw, pd_, pl = goals.outcome_probs(dr, params)
                grid = dixoncoles.scoreline_grid(lh, la, 0.0)  # for scoreline only
            fx.update(played=False, xg_home=round(lh, 2), xg_away=round(la, 2),
                      p_home=round(pw, 4), p_draw=round(pd_, 4), p_away=round(pl, 4),
                      top_scores=_top_scores(grid, 3))
        fixtures.append(fx)

    # ---- knockout fixtures: played (scoreline) + upcoming-known (model W/D/L) ----
    # Upcoming knockout matchups whose teams are already determined get a neutral-
    # venue model prediction (same analytic path as group fixtures) so that
    # append_history can lock a pre-kickoff prediction — giving the knockout stage
    # the same model-vs-market track record as the group stage. Played knockout
    # matches carry no live model probs here: today's model has already absorbed
    # their result, so a pre-kickoff prediction for them comes only from the
    # point-in-time backfill (scripts/backfill_ko_predictions.py), never from this
    # settled-bracket run.
    ko_fixtures = list(ko_results)  # played, with played=True + scoreline
    group_end = fmt["dates"]["group_stage"][-10:]
    upcoming = results[(results.tournament == "FIFA World Cup")
                       & (results.date >= "2026-06-01")
                       & results.home_score.isna()]
    for _, r in upcoming.iterrows():
        h, a = r.home_team, r.away_team
        if h not in team_group or a not in team_group:
            continue  # unresolved bracket slot (placeholder team) — skip
        same_group = team_group[h] == team_group[a]
        if same_group and r.date <= group_end:
            continue  # an unplayed group fixture, not a knockout tie
        probs = sim.matchup_probs(h, a, 0.0)  # knockouts are neutral
        ko_fixtures.append({"date": r.date, "home": h, "away": a, "played": False,
                            "p_home": probs["p_home"], "p_draw": probs["p_draw"],
                            "p_away": probs["p_away"], "xg_home": probs["xg_home"],
                            "xg_away": probs["xg_away"]})
    ko_fixtures.sort(key=lambda x: (x["date"], x["home"]))

    payload = {
        "generated": as_of, "as_of": as_of, "model": model_name,
        "n_sims": out["n_sims"], "params": params,
        "n_played_group": len(played_group), "n_played_ko": len(ko_winners),
        "ratings": {t: round(ratings[t], 1) for t in teams48},
        "advancement": out["probs"], "group_fixtures": fixtures,
        "knockout_fixtures": ko_fixtures,
    }
    with open(ROOT / "output/forecast_latest.json", "w") as f:
        json.dump(payload, f, indent=1)
    print("Wrote output/forecast_latest.json — top 8:")
    rows = sorted(out["probs"].items(), key=lambda kv: -kv[1]["champion"])
    for i, (t, p) in enumerate(rows[:8], 1):
        print(f"  {i}. {t:15s} champion {p['champion']:.1%}")


if __name__ == "__main__":
    main()
