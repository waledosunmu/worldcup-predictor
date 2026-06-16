"""Golden Boot (top scorer) race for the 2026 World Cup.

Distributes each team's model-expected tournament goals among its recent scorers by
goal share (wcpred.scorers), then Poisson-simulates the race. A team's expected
remaining goals = (its average expected goals per match) x (expected remaining
matches = unplayed group fixtures + sum of its reach-round probabilities). Already-
scored 2026 goals are a fixed starting tally.

Inputs : data/raw/goalscorers.csv, output/forecast_latest.json
Output : output/golden_boot.{json,md}
Usage  : python scripts/run_golden_boot.py [--sims 50000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpred import scorers

KO_ROUNDS = ["reach_r32", "reach_r16", "reach_qf", "reach_sf", "reach_final"]
TOP_N = 25
MAX_CONTENDERS = 150


def team_expected_future_goals(forecast: dict) -> dict[str, float]:
    """Per team: avg expected goals/match x expected remaining matches."""
    adv = forecast["advancement"]
    # average expected goals per match from this team's UNPLAYED group fixtures
    xg_sum: dict[str, float] = {}
    xg_n: dict[str, int] = {}
    for fx in forecast["group_fixtures"]:
        if fx.get("played") or "xg_home" not in fx:
            continue
        for team, xg in ((fx["home"], fx["xg_home"]), (fx["away"], fx["xg_away"])):
            xg_sum[team] = xg_sum.get(team, 0.0) + xg
            xg_n[team] = xg_n.get(team, 0) + 1
    avg_xg = {t: xg_sum[t] / xg_n[t] for t in xg_sum}
    global_mean = float(np.mean(list(avg_xg.values()))) if avg_xg else 1.3

    out = {}
    for team in adv:
        a = avg_xg.get(team, global_mean)
        future_matches = xg_n.get(team, 0) + sum(adv[team][r] for r in KO_ROUNDS)
        out[team] = a * future_matches
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=26)
    args = ap.parse_args()

    goalscorers = pd.read_csv(ROOT / "data/raw/goalscorers.csv")
    forecast = json.load(open(ROOT / "output/forecast_latest.json"))
    as_of = forecast["as_of"]
    teams48 = set(forecast["advancement"])

    shares = scorers.recent_goal_shares(goalscorers, as_of)
    current = scorers.current_goals(goalscorers, as_of)
    team_future = team_expected_future_goals(forecast)

    contenders = []
    for team in teams48:
        tfg = team_future.get(team, 0.0)
        for scorer, share in shares.get(team, {}).items():
            contenders.append({"scorer": scorer, "team": team,
                               "current": current.get(scorer, 0),
                               "exp_future": share * tfg})
    # keep the realistic field: top by expected total goals
    contenders.sort(key=lambda c: -(c["current"] + c["exp_future"]))
    contenders = contenders[:MAX_CONTENDERS]

    rng = np.random.default_rng(args.seed)
    ranked = scorers.simulate(contenders, rng, n_sims=args.sims)
    top = ranked[:TOP_N]

    out = {
        "as_of": as_of, "generated": as_of, "n_sims": args.sims,
        "note": ("Goal-share model on martj42 goalscorers.csv (CC0); contenders are "
                 "recent scorers for 2026 finalists, not official squads. Own goals "
                 "excluded, penalties counted."),
        "players": top,
    }
    (ROOT / "output").mkdir(exist_ok=True)
    with open(ROOT / "output/golden_boot.json", "w") as f:
        json.dump(out, f, indent=1)

    lines = ["# Golden Boot race — 2026 World Cup", "",
             f"As of {as_of}. {out['note']}", "",
             "| # | Player | Team | Goals so far | Exp. total | P(Golden Boot) |",
             "|--:|---|---|--:|--:|--:|"]
    for i, p in enumerate(top, 1):
        lines.append(f"| {i} | {p['scorer']} | {p['team']} | {p['current']} | "
                     f"{p['exp_goals']:.1f} | {p['p_win']:.1%} |")
    (ROOT / "output/golden_boot.md").write_text("\n".join(lines) + "\n")

    print(f"Golden Boot race as of {as_of} ({len(contenders)} contenders, {args.sims} sims):")
    for i, p in enumerate(top[:10], 1):
        print(f"  {i:2d}. {p['scorer']:22s} {p['team']:15s} "
              f"now {p['current']}  exp {p['exp_goals']:.1f}  P(win) {p['p_win']:.1%}")
    print("Wrote output/golden_boot.{json,md}")


if __name__ == "__main__":
    main()
