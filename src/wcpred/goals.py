"""Elo-difference -> expected goals (Poisson) model.

lambda_home = exp(a + b * dr/400),  lambda_away = exp(a - b * dr/400)
where dr is the pre-match Elo difference including home advantage. Fit by
joint Poisson maximum likelihood. This is the simplest member of the
Groll-style "ability -> expected goals -> Poisson" family; Dixon-Coles
low-score adjustment and covariate models can replace it behind the same
interface later.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import skellam


def fit(dr: np.ndarray, home_goals: np.ndarray, away_goals: np.ndarray) -> dict:
    x = dr / 400.0

    def nll(params):
        a, b = params
        lh = np.exp(a + b * x)
        la = np.exp(a - b * x)
        return -(np.sum(home_goals * np.log(lh) - lh)
                 + np.sum(away_goals * np.log(la) - la))

    res = minimize(nll, x0=[0.2, 0.5], method="Nelder-Mead")
    a, b = res.x
    return {"a": float(a), "b": float(b), "nll": float(res.fun), "n": int(len(x))}


def expected_goals(dr: float, params: dict) -> tuple[float, float]:
    x = dr / 400.0
    return (float(np.exp(params["a"] + params["b"] * x)),
            float(np.exp(params["a"] - params["b"] * x)))


def outcome_probs(dr: float, params: dict) -> tuple[float, float, float]:
    """P(win, draw, loss) for the dr-advantaged side over 90 minutes,
    from the Skellam distribution of the goal difference."""
    lh, la = expected_goals(dr, params)
    p_draw = float(skellam.pmf(0, lh, la))
    p_loss = float(skellam.cdf(-1, lh, la))
    return 1.0 - p_draw - p_loss, p_draw, p_loss
