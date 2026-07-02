"""Point-in-time match model shared by the tournament forecast and the knockout
backfill, so both derive predictions from the SAME as-of-date model: Elo ratings,
refit goals parameters, and current form, ALL computed from matches strictly
before `as_of`.

This is the leak-free contract (CLAUDE.md cardinal rule #1): a prediction for a
match on date D uses only matches dated < D. The forward forecast and any
backfill call one path here so a params-refit leak can't slip into one but not
the other.
"""
from __future__ import annotations

import pandas as pd

from . import covariates, dixoncoles, elo, goals

FIT_WINDOW_YEARS = 28
FORM_WINDOW = 5


def build_asof_model(results: pd.DataFrame, as_of: str, model: str = "hybrid") -> dict:
    """Elo ratings + refit goals params + team form as of `as_of` (exclusive).

    Mirrors the model construction in scripts/run_forecast.py so the live forecast
    and the knockout backfill stay one methodology. Returns a bundle whose
    ratings/params/team_form/hybrid feed straight into simulate.Simulator, whose
    ``matchup_probs`` then yields the analytic W/D/L.

    Every input is cut at ``date < as_of`` (elo.compute_ratings honours as_of
    internally), so the output is invariant to any match dated on/after as_of.
    """
    hist = elo.rating_history_frame(results[results.date < as_of])
    hist = covariates.build_form_frame(hist, window=FORM_WINDOW)
    ratings = elo.compute_ratings(results, as_of=as_of)
    fit_df = hist[(hist.date >= f"{int(as_of[:4]) - FIT_WINDOW_YEARS}-01-01")
                  & (hist.tournament != "Friendly")]
    dr = fit_df.dr.to_numpy()
    hg = fit_df.home_score.to_numpy()
    ag = fit_df.away_score.to_numpy()
    hybrid = model == "hybrid"
    if hybrid:
        Z = covariates._design(fit_df, ["form_diff"])
        params = dixoncoles.fit_hybrid(dr, hg, ag, Z=Z, covariate_names=["form_diff"],
                                       x0=goals.fit(dr, hg, ag))
        team_form = covariates.current_form(hist, window=FORM_WINDOW, as_of=as_of)
        model_name = "elo-poisson-form-dc-conditional"
    else:
        params = goals.fit(dr, hg, ag)
        team_form = None
        model_name = "elo-poisson-v0-conditional"
    return {"ratings": ratings, "params": params, "team_form": team_form,
            "hybrid": hybrid, "model_name": model_name}
