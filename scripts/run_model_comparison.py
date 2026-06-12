"""Head-to-head model comparison: v0 independent-Poisson vs Dixon-Coles vs
Dixon-Coles+time-decay vs covariate-augmented (Groll-style), on World Cups
2006-2022 with strict point-in-time discipline.

Every model is fit ONLY on competitive internationals played before each
edition's opening day (28-year window, non-friendly, matching the live
forecast), then scores that edition's 64 matches as 3-way W/D/L forecasts.

Reported per model:
  - in-sample fit: total log-likelihood on the pre-2006 fit window (one number,
    on a common window so the LL is comparable across models)
  - out-of-sample: pooled log loss / Brier / RPS over all 320 WC matches

The time-decay half-life is selected purely OUT-OF-SAMPLE (grid -> best pooled
log loss), because the weighted in-sample LL is not comparable across half-lives.

Writes output/model_comparison.{md,json}. Reuses wcpred.metrics and the same
fit windows as scripts/run_backtest.py.

Usage: python scripts/run_model_comparison.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpred import covariates, dixoncoles, elo, goals, metrics

EDITIONS = [2006, 2010, 2014, 2018, 2022]
FIT_WINDOW_YEARS = 28
ET_FACTOR = 0.33
FORM_WINDOW = 5
DECAY_HALF_LIVES = [0, 2, 4, 8, 16]  # years; 0 = no decay (uniform weights)


def outcome_index(hs, as_):
    return 0 if hs > as_ else (2 if hs < as_ else 1)


def grid_wdl(grid):
    n = grid.shape[0]
    i = np.arange(n)
    hh, aa = np.meshgrid(i, i, indexing="ij")
    return (float(grid[hh > aa].sum()), float(np.trace(grid)), float(grid[hh < aa].sum()))


def poisson_probs(dr, params, knockout):
    pw, pd_, pl = goals.outcome_probs(dr, params)
    if not knockout:
        return np.array([pw, pd_, pl])
    from scipy.stats import skellam
    lh, la = goals.expected_goals(dr, params)
    pd_et = float(skellam.pmf(0, lh * ET_FACTOR, la * ET_FACTOR))
    pl_et = float(skellam.cdf(-1, lh * ET_FACTOR, la * ET_FACTOR))
    return np.array([pw + pd_ * (1 - pd_et - pl_et), pd_ * pd_et, pl + pd_ * pl_et])


def dc_probs(dr, params, knockout):
    pw, pd_, pl = dixoncoles.outcome_probs(dr, params)
    if not knockout:
        return np.array([pw, pd_, pl])
    lh, la = goals.expected_goals(dr, params)
    et = dixoncoles.scoreline_grid(lh * ET_FACTOR, la * ET_FACTOR, params["rho"])
    pw_et, pd_et, pl_et = grid_wdl(et)
    return np.array([pw + pd_ * pw_et, pd_ * pd_et, pl + pd_ * pl_et])


def cov_probs(dr, z, params, knockout):
    """Covariate model W/D/L via independent-Poisson/Skellam on augmented means."""
    from scipy.stats import skellam
    lh, la = covariates.expected_goals(dr, z, params)
    pd_ = float(skellam.pmf(0, lh, la))
    pl = float(skellam.cdf(-1, lh, la))
    pw = 1.0 - pd_ - pl
    if not knockout:
        return np.array([pw, pd_, pl])
    pd_et = float(skellam.pmf(0, lh * ET_FACTOR, la * ET_FACTOR))
    pl_et = float(skellam.cdf(-1, lh * ET_FACTOR, la * ET_FACTOR))
    return np.array([pw + pd_ * (1 - pd_et - pl_et), pd_ * pd_et, pl + pd_ * pl_et])


def hybrid_probs(dr, z, params, knockout):
    """Combined covariate + Dixon-Coles W/D/L: covariate-augmented means feeding
    the DC-corrected joint scoreline grid."""
    lh, la = covariates.expected_goals(dr, z, params)
    grid = dixoncoles.scoreline_grid(lh, la, params["rho"])
    pw, pd_, pl = grid_wdl(grid)
    if not knockout:
        return np.array([pw, pd_, pl])
    et = dixoncoles.scoreline_grid(lh * ET_FACTOR, la * ET_FACTOR, params["rho"])
    pw_et, pd_et, pl_et = grid_wdl(et)
    return np.array([pw + pd_ * pw_et, pd_ * pd_et, pl + pd_ * pl_et])


def score(probs, o):
    return metrics.log_loss(probs, o), metrics.brier(probs, o), metrics.rps(probs, o)


def main():
    results = pd.read_csv(ROOT / "data/raw/results.csv")
    print("Replaying Elo history + building point-in-time form frame...")
    hist = elo.rating_history_frame(results)
    hist = covariates.build_form_frame(hist, window=FORM_WINDOW)
    hist = covariates.add_host_covariate(hist)
    wc = hist[hist.tournament == "FIFA World Cup"].copy()
    wc["year"] = wc.date.str[:4].astype(int)

    models = ["v0_poisson", "dixon_coles", "covar_form", "covar_form_host",
              "hybrid_form_dc"]
    models += [f"dc_decay_hl{hl}" for hl in DECAY_HALF_LIVES if hl > 0]
    oos = {m: {"logloss": [], "brier": [], "rps": []} for m in models}
    per_edition = []
    insample = {m: 0.0 for m in models}
    fitted = {m: [] for m in models}

    for year in EDITIONS:
        ed = wc[wc.year == year].sort_values("date").reset_index(drop=True)
        opening = ed.date.min()
        fit = hist[(hist.date < opening)
                   & (hist.date >= f"{year - FIT_WINDOW_YEARS}-01-01")
                   & (hist.tournament != "Friendly")]
        dr = fit.dr.to_numpy()
        hg = fit.home_score.to_numpy()
        ag = fit.away_score.to_numpy()

        p0 = goals.fit(dr, hg, ag)
        pdc = dixoncoles.fit(dr, hg, ag, x0=p0)
        Zf = covariates._design(fit, ["form_diff"])
        Zfh = covariates._design(fit, ["form_diff", "host"])
        pcf = covariates.fit(dr, hg, ag, Z=Zf, covariate_names=["form_diff"], x0=p0)
        pcfh = covariates.fit(dr, hg, ag, Z=Zfh,
                              covariate_names=["form_diff", "host"], x0=p0)
        phyb = dixoncoles.fit_hybrid(dr, hg, ag, Z=Zf,
                                     covariate_names=["form_diff"], x0=p0)
        params = {"v0_poisson": p0, "dixon_coles": pdc,
                  "covar_form": pcf, "covar_form_host": pcfh,
                  "hybrid_form_dc": phyb}
        # time-decayed Dixon-Coles fits (weights by half-life relative to opening)
        for hl in DECAY_HALF_LIVES:
            if hl == 0:
                continue
            w = dixoncoles.time_decay_weights(fit.date.to_numpy(), opening, hl)
            params[f"dc_decay_hl{hl}"] = dixoncoles.fit(dr, hg, ag, weights=w, x0=p0)

        for m in models:
            insample[m] += -params[m]["nll"]  # nll is weighted for decay models
            fitted[m].append({"edition": year, **{k: v for k, v in params[m].items()
                              if k in ("a", "b", "rho", "beta", "n", "covariates")}})

        ed_m = {m: {"logloss": [], "brier": [], "rps": []} for m in models}
        for i, r in ed.iterrows():
            knockout = i >= 48
            o = outcome_index(r.home_score, r.away_score)
            z = {"form_diff": r.form_diff, "host": r.host}
            preds = {
                "v0_poisson": poisson_probs(r.dr, p0, knockout),
                "dixon_coles": dc_probs(r.dr, pdc, knockout),
                "covar_form": cov_probs(r.dr, z, pcf, knockout),
                "covar_form_host": cov_probs(r.dr, z, pcfh, knockout),
                "hybrid_form_dc": hybrid_probs(r.dr, z, phyb, knockout),
            }
            for hl in DECAY_HALF_LIVES:
                if hl > 0:
                    preds[f"dc_decay_hl{hl}"] = dc_probs(r.dr, params[f"dc_decay_hl{hl}"], knockout)
            for m, p in preds.items():
                ll, br, rp = score(p, o)
                ed_m[m]["logloss"].append(ll)
                ed_m[m]["brier"].append(br)
                ed_m[m]["rps"].append(rp)

        for m in models:
            for k in oos[m]:
                oos[m][k].extend(ed_m[m][k])
            per_edition.append({"edition": year, "model": m,
                                **{k: float(np.mean(v)) for k, v in ed_m[m].items()}})
        print(f"{year}: v0 RPS {np.mean(ed_m['v0_poisson']['rps']):.4f} | "
              f"DC RPS {np.mean(ed_m['dixon_coles']['rps']):.4f} | "
              f"covar_form RPS {np.mean(ed_m['covar_form']['rps']):.4f} | rho {pdc['rho']:.4f}")

    pooled = {m: {k: float(np.mean(v)) for k, v in oos[m].items()} for m in models}

    # pick best decay half-life by OOS log loss
    decay_models = [m for m in models if m.startswith("dc_decay")]
    best_decay = min(decay_models, key=lambda m: pooled[m]["logloss"]) if decay_models else None

    out = {
        "editions": EDITIONS, "fit_window_years": FIT_WINDOW_YEARS,
        "form_window": FORM_WINDOW, "decay_half_lives_years": DECAY_HALF_LIVES,
        "note": ("In-sample LL is the sum over editions of the pre-opening fit "
                 "log-likelihood (weighted for decay models, so cross-decay LL is "
                 "NOT comparable -- decay is selected by OOS log loss only). "
                 "OOS metrics are pooled over all 320 WC matches, lower is better."),
        "in_sample_loglik_sum": {m: float(insample[m]) for m in models},
        "oos_pooled": pooled,
        "best_decay_by_oos_logloss": best_decay,
        "per_edition": per_edition,
        "fitted_params": fitted,
        "covariate_spec": covariates.COVARIATE_SPEC,
    }
    (ROOT / "output").mkdir(exist_ok=True)
    with open(ROOT / "output/model_comparison.json", "w") as f:
        json.dump(out, f, indent=1)

    order = ["v0_poisson", "dixon_coles", "covar_form", "covar_form_host",
             "hybrid_form_dc"] + decay_models
    lines = [
        "# Model comparison: v0 Poisson vs Dixon-Coles vs covariate (Groll-style)",
        "",
        "Point-in-time: each model fit only on competitive (non-friendly) "
        "internationals before each edition's opening day, 28-year window.",
        "Out-of-sample = the 64 matches of each World Cup 2006-2022 (320 total), "
        "scored as 3-way W/D/L. Lower is better for all metrics.",
        "",
        "## Out-of-sample, pooled over 320 World Cup matches",
        "",
        "| Model | Log loss | Brier | RPS | In-sample sum log-lik |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in order:
        p = pooled[m]
        # decay models carry a *weighted* LL on a different scale -> not
        # comparable to the unweighted rows; show n/a rather than mislead.
        ll_cell = "n/a (weighted)" if m.startswith("dc_decay") else f"{insample[m]:.1f}"
        lines.append(f"| {m} | {p['logloss']:.4f} | {p['brier']:.4f} | {p['rps']:.4f} "
                     f"| {ll_cell} |")
    lines += [
        "",
        f"Best time-decay half-life by OOS log loss: "
        f"{best_decay or 'n/a'}.",
        "",
        "The in-sample column sums each edition's pre-opening fit-window "
        "log-likelihood. It is comparable across the non-decay models (same "
        "data, more parameters -> higher is better) but NOT across decay "
        "half-lives (different weights change the LL scale).",
        "",
        "## Per edition, RPS",
        "",
        "| Edition | " + " | ".join(order) + " |",
        "|---|" + "---:|" * len(order),
    ]
    for year in EDITIONS:
        vals = [next(r for r in per_edition
                     if r["edition"] == year and r["model"] == m)["rps"] for m in order]
        lines.append(f"| {year} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")

    lines += [
        "",
        "## Covariate availability (Groll et al. full hybrid)",
        "",
        "| Covariate | Available here | Notes / data needed |",
        "|---|:--:|---|",
    ]
    for k, v in covariates.COVARIATE_SPEC.items():
        mark = "yes" if v["available"] else "no"
        lines.append(f"| {k} | {mark} | {v['desc']}; source: {v['source']} |")
    lines += [
        "",
        "Covariates marked 'no' require external data not shipped in this repo "
        "and are scaffolded (COVARIATE_SPEC + the symmetric design-matrix "
        "interface in covariates.fit) but never fabricated.",
    ]
    (ROOT / "output/model_comparison.md").write_text("\n".join(lines) + "\n")

    print("\nOOS pooled (lower is better):")
    for m in order:
        p = pooled[m]
        print(f"  {m:18s} logloss {p['logloss']:.4f}  brier {p['brier']:.4f}  rps {p['rps']:.4f}")
    print(f"best decay half-life (OOS logloss): {best_decay}")
    print("Wrote output/model_comparison.{json,md}")


if __name__ == "__main__":
    main()
