"""Opening-day forecast for the 2026 World Cup.

Pipeline: point-in-time Elo over full history -> Poisson goals model fit on
competitive internationals -> 2026-format Monte Carlo -> JSON + markdown report.

Usage: python scripts/run_opening_forecast.py [--sims 100000]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpred import elo, goals
from wcpred.simulate import Simulator, load_format

AS_OF = "2026-06-11"
FIT_FROM = "1998-01-01"

# martj42 name -> eloratings.net name, where they differ (48 squads only)
ELO_NAME_MAP = {"Czech Republic": "Czechia"}


def official_elo(raw_dir: Path) -> dict[str, float]:
    codes = {}
    for line in (raw_dir / "elo_teams.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            codes[parts[0]] = parts[1]
    ratings = {}
    for line in (raw_dir / "elo_world.tsv").read_text().splitlines():
        p = line.split("\t")
        if len(p) >= 4 and p[2] in codes:
            ratings[codes[p[2]]] = float(p[3])
    return ratings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=26)
    args = ap.parse_args()

    t0 = time.time()
    results = pd.read_csv(ROOT / "data/raw/results.csv")
    fmt = load_format(ROOT / "data/format_2026.json")
    teams48 = [t for g in sorted(fmt["groups"]) for t in fmt["groups"][g]]

    print("== Elo: replaying full history with point-in-time ratings ==")
    hist = elo.rating_history_frame(results[results.date < AS_OF])
    ratings = elo.compute_ratings(results, as_of=AS_OF)
    missing = [t for t in teams48 if t not in ratings]
    if missing:
        raise SystemExit(f"teams missing from Elo engine: {missing}")

    print("== Sanity check vs official eloratings.net ==")
    official = official_elo(ROOT / "data/raw")
    pairs = []
    for t in teams48:
        name = ELO_NAME_MAP.get(t, t)
        if name in official:
            pairs.append((ratings[t], official[name]))
    ours, theirs = map(np.array, zip(*pairs))
    corr = np.corrcoef(ours, theirs)[0, 1]
    print(f"   {len(pairs)}/48 teams matched; correlation ours vs official: {corr:.3f}")

    print("== Fitting Poisson goals model ==")
    fit_df = hist[(hist.date >= FIT_FROM) & (hist.tournament != "Friendly")]
    params = goals.fit(fit_df.dr.to_numpy(),
                       fit_df.home_score.to_numpy(), fit_df.away_score.to_numpy())
    print(f"   a={params['a']:.4f} b={params['b']:.4f} on n={params['n']} matches")

    print(f"== Simulating {args.sims:,} tournaments ==")
    sim = Simulator(fmt, ratings, params, seed=args.seed)
    out = sim.run(args.sims)
    print(f"   done in {time.time()-t0:.0f}s total")

    # analytic per-fixture probabilities for the group stage
    fixtures = []
    for g, home, away, lh, la in sim.group_fixtures:
        dr = ratings[home] - ratings[away] + (100.0 if home in {"Mexico", "Canada", "United States"} else 0.0)
        pw, pd_, pl = goals.outcome_probs(dr, params)
        fixtures.append({"group": g, "home": home, "away": away,
                         "xg_home": round(lh, 2), "xg_away": round(la, 2),
                         "p_home": round(pw, 4), "p_draw": round(pd_, 4),
                         "p_away": round(pl, 4),
                         "elo_home": round(ratings[home]), "elo_away": round(ratings[away])})

    payload = {
        "generated": AS_OF, "model": "elo-poisson-v0", "n_sims": out["n_sims"],
        "params": params, "elo_correlation_official": round(float(corr), 4),
        "ratings": {t: round(ratings[t], 1) for t in teams48},
        "advancement": out["probs"], "group_fixtures": fixtures,
    }
    (ROOT / "output").mkdir(exist_ok=True)
    with open(ROOT / "output/forecast_opening.json", "w") as f:
        json.dump(payload, f, indent=1)

    # markdown report
    team_group = {t: g for g in fmt["groups"] for t in fmt["groups"][g]}
    rows = sorted(out["probs"].items(), key=lambda kv: -kv[1]["champion"])
    lines = [
        "# World Cup 2026 — Opening-Day Forecast (Elo–Poisson v0)",
        "",
        f"Generated {AS_OF} from {out['n_sims']:,} Monte Carlo simulations. "
        f"Ratings: in-house World-Football-Elo replication (corr. {corr:.3f} with eloratings.net). "
        f"Goals model: Poisson MLE on {params['n']:,} competitive internationals since {FIT_FROM[:4]}.",
        "",
        "| Rk | Team | Grp | Elo | R32 | R16 | QF | SF | Final | Champion |",
        "|---:|------|-----|----:|----:|----:|---:|---:|------:|---------:|",
    ]
    for i, (t, p) in enumerate(rows[:20], 1):
        lines.append(
            f"| {i} | {t} | {team_group[t]} | {ratings[t]:.0f} "
            f"| {p['reach_r32']:.0%} | {p['reach_r16']:.0%} | {p['reach_qf']:.0%} "
            f"| {p['reach_sf']:.0%} | {p['reach_final']:.0%} | {p['champion']:.1%} |")
    op = fixtures[0]
    for f_ in fixtures:
        if f_["home"] == "Mexico" and f_["away"] == "South Africa":
            op = f_
    lines += ["", "## Opening match",
              f"**{op['home']} vs {op['away']}** — {op['p_home']:.0%} / {op['p_draw']:.0%} / "
              f"{op['p_away']:.0%} (xG {op['xg_home']}–{op['xg_away']})"]
    (ROOT / "output/forecast_opening.md").write_text("\n".join(lines) + "\n")
    print("Wrote output/forecast_opening.json and output/forecast_opening.md")
    for i, (t, p) in enumerate(rows[:10], 1):
        print(f"  {i:2d}. {t:15s} champion {p['champion']:.1%}  final {p['reach_final']:.0%}")


if __name__ == "__main__":
    main()
