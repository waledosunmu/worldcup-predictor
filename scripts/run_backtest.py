"""Backtest model candidates on World Cups 2006-2022 with point-in-time discipline.

For each edition: Elo ratings and goals-model parameters are computed using only
matches played BEFORE that tournament's opening day (fit window = 28 years, the
same the 2026 forecast uses). Each of the 64 matches is scored as a 3-way
probabilistic forecast (home/draw/away, where knockout 'draw' = went to
penalties, matching how the dataset records scores: ET included, shootouts not).

Models:
  elo_poisson     - Skellam probs from the Elo->Poisson goals model (KO: ET-aware)
  elo_expectancy  - Elo expected-result + prior-WC draw share (classic baseline)
  climatology     - constant prior-WC home/draw/away shares (floor)

Also: group-qualification check — simulate each group 10k times under legacy
(pre-2026) tiebreakers and score P(top-2) against the actual R16 qualifiers.

Usage: python scripts/run_backtest.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpred import dixoncoles, elo, goals, metrics

EDITIONS = [2006, 2010, 2014, 2018, 2022]
FIT_WINDOW_YEARS = 28
ET_FACTOR = 0.33
N_GROUP_SIMS = 10_000
MODELS = ["elo_poisson", "elo_poisson_dc", "elo_expectancy", "climatology"]


def outcome_index(hs: int, as_: int) -> int:
    return 0 if hs > as_ else (2 if hs < as_ else 1)


def elo_poisson_probs(dr: float, params: dict, knockout: bool) -> np.ndarray:
    pw, pd_, pl = goals.outcome_probs(dr, params)
    if not knockout:
        return np.array([pw, pd_, pl])
    lh, la = goals.expected_goals(dr, params)
    from scipy.stats import skellam
    pd_et = float(skellam.pmf(0, lh * ET_FACTOR, la * ET_FACTOR))
    pl_et = float(skellam.cdf(-1, lh * ET_FACTOR, la * ET_FACTOR))
    pw_et = 1.0 - pd_et - pl_et
    return np.array([pw + pd_ * pw_et, pd_ * pd_et, pl + pd_ * pl_et])


def elo_poisson_dc_probs(dr: float, params: dict, knockout: bool) -> np.ndarray:
    """Dixon-Coles-corrected W/D/L from the truncated joint scoreline grid.
    Knockout 'draw' is split into ET-resolved win/draw/loss like the Poisson
    path, using the DC grid at ET-scaled means for the still-level branch."""
    pw, pd_, pl = dixoncoles.outcome_probs(dr, params)
    if not knockout:
        return np.array([pw, pd_, pl])
    lh, la = goals.expected_goals(dr, params)
    et = dixoncoles.scoreline_grid(lh * ET_FACTOR, la * ET_FACTOR, params["rho"])
    n = et.shape[0]
    i = np.arange(n)
    hh, aa = np.meshgrid(i, i, indexing="ij")
    pw_et = float(et[hh > aa].sum())
    pd_et = float(np.trace(et))
    pl_et = float(et[hh < aa].sum())
    return np.array([pw + pd_ * pw_et, pd_ * pd_et, pl + pd_ * pl_et])


def elo_expectancy_probs(dr: float, draw_share: float) -> np.ndarray:
    we = elo.expectancy(dr)
    pw = max(we - draw_share / 2.0, 1e-4)
    pl = max(1.0 - we - draw_share / 2.0, 1e-4)
    p = np.array([pw, draw_share, pl])
    return p / p.sum()


def legacy_rank_key(t, stats, h2h_stats, noise):
    """Pre-2026 rules: overall pts/GD/GF first, then h2h, then lots (noise)."""
    pts, gf, ga = stats[t]
    hp, hgf, hga = h2h_stats.get(t, (0, 0, 0))
    return (pts, gf - ga, gf, hp, hgf - hga, hgf, noise[t])


def simulate_group_legacy(fixture_lams, teams, rng):
    """fixture_lams: [(home, away, lh, la)] x6. Returns P(top2) per team."""
    lam = np.array([[f[2], f[3]] for f in fixture_lams])
    draws = rng.poisson(lam, size=(N_GROUP_SIMS, 6, 2))
    top2 = {t: 0 for t in teams}
    for s in range(N_GROUP_SIMS):
        stats = {t: [0, 0, 0] for t in teams}
        results = {}
        for (home, away, _, _), (gh, ga) in zip(fixture_lams, draws[s]):
            gh, ga = int(gh), int(ga)
            results[(home, away)] = (gh, ga)
            stats[home][1] += gh; stats[home][2] += ga
            stats[away][1] += ga; stats[away][2] += gh
            if gh > ga:
                stats[home][0] += 3
            elif ga > gh:
                stats[away][0] += 3
            else:
                stats[home][0] += 1; stats[away][0] += 1
        # h2h only needed within tied clusters; legacy rules apply it after
        # overall pts/GD/GF, and full h2h mini-tables among exact ties are a
        # rare refinement - random noise breaks residual ties (proxy for lots)
        noise = {t: rng.random() for t in teams}
        h2h_stats = {}
        order = sorted(teams, key=lambda t: legacy_rank_key(t, stats, h2h_stats, noise),
                       reverse=True)
        top2[order[0]] += 1
        top2[order[1]] += 1
    return {t: c / N_GROUP_SIMS for t, c in top2.items()}


def main():
    results = pd.read_csv(ROOT / "data/raw/results.csv")
    print("Replaying Elo history (point-in-time pre-match ratings)...")
    hist = elo.rating_history_frame(results)
    wc = hist[hist.tournament == "FIFA World Cup"].copy()
    wc["year"] = wc.date.str[:4].astype(int)

    rows, qual_rows = [], []
    per_match = {m: {"logloss": [], "brier": [], "rps": []} for m in MODELS}

    for year in EDITIONS:
        ed = wc[wc.year == year].sort_values("date").reset_index(drop=True)
        assert len(ed) == 64, f"{year}: expected 64 matches, got {len(ed)}"
        opening = ed.date.min()

        fit = hist[(hist.date < opening)
                   & (hist.date >= f"{year - FIT_WINDOW_YEARS}-01-01")
                   & (hist.tournament != "Friendly")]
        params = goals.fit(fit.dr.to_numpy(),
                           fit.home_score.to_numpy(), fit.away_score.to_numpy())
        # Dixon-Coles: add rho by MLE on the same pre-opening window (no decay
        # here; decay is selected out-of-sample in the separate sweep below).
        params_dc = dixoncoles.fit(fit.dr.to_numpy(), fit.home_score.to_numpy(),
                                   fit.away_score.to_numpy(), x0=params)

        prior_wc = hist[(hist.tournament == "FIFA World Cup") & (hist.date < opening)]
        prior_out = np.array([outcome_index(h, a) for h, a in
                              zip(prior_wc.home_score, prior_wc.away_score)])
        clim = np.array([(prior_out == k).mean() for k in range(3)])
        draw_share = clim[1]

        ed_metrics = {m: {"logloss": [], "brier": [], "rps": []} for m in MODELS}
        for i, r in ed.iterrows():
            knockout = i >= 48
            o = outcome_index(r.home_score, r.away_score)
            preds = {
                "elo_poisson": elo_poisson_probs(r.dr, params, knockout),
                "elo_poisson_dc": elo_poisson_dc_probs(r.dr, params_dc, knockout),
                "elo_expectancy": elo_expectancy_probs(r.dr, draw_share),
                "climatology": clim.copy(),
            }
            for m, p in preds.items():
                ed_metrics[m]["logloss"].append(metrics.log_loss(p, o))
                ed_metrics[m]["brier"].append(metrics.brier(p, o))
                ed_metrics[m]["rps"].append(metrics.rps(p, o))

        for m in MODELS:
            for k in per_match[m]:
                per_match[m][k].extend(ed_metrics[m][k])
            rows.append({"edition": year, "model": m,
                         **{k: float(np.mean(v)) for k, v in ed_metrics[m].items()}})

        # --- group-qualification eval (elo_poisson) ---
        rng = np.random.default_rng(year)
        group_matches = ed.iloc[:48]
        ko_teams = set(ed.iloc[48:56].home_team) | set(ed.iloc[48:56].away_team)
        adj = {}
        for _, r in group_matches.iterrows():
            adj.setdefault(r.home_team, set()).add(r.away_team)
            adj.setdefault(r.away_team, set()).add(r.home_team)
        seen, briers, p_qual = set(), [], []
        for t in sorted(adj):
            if t in seen:
                continue
            teams = sorted({t} | adj[t])
            seen |= set(teams)
            fl = []
            for _, r in group_matches.iterrows():
                if r.home_team in teams and r.away_team in teams:
                    lh, la = goals.expected_goals(r.dr, params)
                    fl.append((r.home_team, r.away_team, lh, la))
            ptop2 = simulate_group_legacy(fl, teams, rng)
            for team, p in ptop2.items():
                actual = 1.0 if team in ko_teams else 0.0
                briers.append((p - actual) ** 2)
                if actual:
                    p_qual.append(p)
        qual_rows.append({"edition": year,
                          "mean_p_actual_qualifiers": float(np.mean(p_qual)),
                          "qual_brier": float(np.mean(briers)),
                          "fit_n": params["n"], "fit_a": params["a"], "fit_b": params["b"],
                          "fit_rho": params_dc["rho"]})
        print(f"{year}: elo_poisson RPS {np.mean(ed_metrics['elo_poisson']['rps']):.4f} | "
              f"P(actual qualifiers) {np.mean(p_qual):.2f} | fit n={params['n']}")

    pooled = [{"edition": "ALL", "model": m,
               **{k: float(np.mean(v)) for k, v in per_match[m].items()}}
              for m in MODELS]

    out = {"editions": EDITIONS, "fit_window_years": FIT_WINDOW_YEARS,
           "per_edition": rows, "pooled": pooled, "group_qualification": qual_rows}
    (ROOT / "output").mkdir(exist_ok=True)
    with open(ROOT / "output/backtest_2006_2022.json", "w") as f:
        json.dump(out, f, indent=1)

    lines = ["# Backtest: World Cups 2006-2022 (point-in-time, 64 matches each)", "",
             "Knockout 'draw' = match went to penalties. Lower is better for all metrics.",
             "", "## Pooled over all 320 matches", "",
             "| Model | Log loss | Brier | RPS |", "|---|---:|---:|---:|"]
    for p in pooled:
        lines.append(f"| {p['model']} | {p['logloss']:.4f} | {p['brier']:.4f} | {p['rps']:.4f} |")
    lines += ["", "## Per edition (RPS)", "",
              "| Edition | " + " | ".join(MODELS) + " |",
              "|---|" + "---:|" * len(MODELS)]
    for year in EDITIONS:
        vals = [next(r for r in rows if r["edition"] == year and r["model"] == m)["rps"]
                for m in MODELS]
        lines.append(f"| {year} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    lines += ["", "## Group-stage qualification (elo_poisson, legacy tiebreakers)", "",
              "| Edition | mean P assigned to actual R16 qualifiers | Brier (qualify) |",
              "|---|---:|---:|"]
    for q in qual_rows:
        lines.append(f"| {q['edition']} | {q['mean_p_actual_qualifiers']:.3f} "
                     f"| {q['qual_brier']:.4f} |")
    (ROOT / "output/backtest_2006_2022.md").write_text("\n".join(lines) + "\n")
    print("\nPooled results:")
    for p in pooled:
        print(f"  {p['model']:15s} logloss {p['logloss']:.4f}  brier {p['brier']:.4f}  "
              f"rps {p['rps']:.4f}")
    print("Wrote output/backtest_2006_2022.{json,md}")


if __name__ == "__main__":
    main()
