"""Dixon-Coles (1997) low-score dependence correction on top of the
Elo-difference -> Poisson goals model in goals.py.

The independent-Poisson model (goals.py) systematically mis-prices the four
low-scoring scorelines 0-0, 1-0, 0-1, 1-1. Dixon & Coles (Applied Statistics
1997) multiply those four joint probabilities by a dependence factor tau(rho):

    tau(0,0) = 1 - lh*la*rho
    tau(1,0) = 1 + la*rho
    tau(0,1) = 1 + lh*rho
    tau(1,1) = 1 - rho
    tau(.,.) = 1            otherwise

with lh, la the Poisson means. rho<0 boosts the 0-0 and 1-1 cells (since
tau(0,0), tau(1,1) > 1 there) and is the sign typically found for football,
where independence under-predicts low-scoring draws. The MLE rho here is
negative and stable across editions, consistent with that. The full
marginal-goals model (a, b) is shared with v0; only
rho is added, fit jointly by maximum likelihood. Optional exponential time-decay
weights w_t = exp(-xi * age_years) down-weight old matches (Dixon-Coles sec. 4).

Everything is built from a truncated joint scoreline grid so the simulator and
the W/D/L scorer see the *corrected joint distribution* (Skellam is only valid
under independence and would silently ignore rho). At rho=0 the grid reproduces
the independent-Poisson probabilities exactly.
"""

import numpy as np
from scipy.optimize import minimize

from .goals import expected_goals

MAX_GOALS = 15  # grid truncation; P(>15 goals one side) is ~0 for football lambdas


def tau(home_goals, away_goals, lh, la, rho):
    """Dixon-Coles low-score adjustment factor (vectorised over goal arrays)."""
    hg = np.asarray(home_goals)
    ag = np.asarray(away_goals)
    t = np.ones(np.broadcast(hg, ag, lh, la).shape, dtype=float)
    t = np.where((hg == 0) & (ag == 0), 1.0 - lh * la * rho, t)
    t = np.where((hg == 1) & (ag == 0), 1.0 + la * rho, t)
    t = np.where((hg == 0) & (ag == 1), 1.0 + lh * rho, t)
    t = np.where((hg == 1) & (ag == 1), 1.0 - rho, t)
    return t


def scoreline_grid(lh: float, la: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """(max_goals+1, max_goals+1) joint P(home=i, away=j) with the DC correction
    applied and renormalised. At rho=0 this is the outer product of the two
    independent Poisson pmfs (truncated)."""
    i = np.arange(max_goals + 1)
    # truncated Poisson pmfs
    log_fact = np.cumsum(np.concatenate(([0.0], np.log(np.arange(1, max_goals + 1)))))
    ph = np.exp(i * np.log(lh) - lh - log_fact)
    pa = np.exp(i * np.log(la) - la - log_fact)
    grid = np.outer(ph, pa)
    hh, aa = np.meshgrid(i, i, indexing="ij")
    grid = grid * tau(hh, aa, lh, la, rho)
    grid = np.clip(grid, 0.0, None)
    return grid / grid.sum()


def fit(dr: np.ndarray, home_goals: np.ndarray, away_goals: np.ndarray,
        weights: np.ndarray | None = None, x0: dict | None = None) -> dict:
    """Joint MLE of (a, b, rho). `weights` optional per-match likelihood weights
    (e.g. exponential time decay). `x0` optionally seeds a/b from a v0 fit.

    The DC tau correction touches only the four low-score cells, so the closed-
    form likelihood of Dixon-Coles is used directly (no grid needed for fitting):
        L = prod_t tau(h,a; lh,la,rho) * Pois(h;lh) * Pois(a;la)
    """
    x = dr / 400.0
    hg = np.asarray(home_goals, dtype=float)
    ag = np.asarray(away_goals, dtype=float)
    w = np.ones_like(hg) if weights is None else np.asarray(weights, dtype=float)
    a0 = (x0 or {}).get("a", 0.2)
    b0 = (x0 or {}).get("b", 0.5)

    def nll(params):
        a, b, rho = params
        lh = np.exp(a + b * x)
        la = np.exp(a - b * x)
        t = tau(hg, ag, lh, la, rho)
        if np.any(t <= 0):
            return 1e12  # invalid rho -> non-positive cell probability
        ll = (hg * np.log(lh) - lh + ag * np.log(la) - la + np.log(t))
        return -float(np.sum(w * ll))

    res = minimize(nll, x0=[a0, b0, 0.05], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
    a, b, rho = res.x
    return {"a": float(a), "b": float(b), "rho": float(rho),
            "nll": float(res.fun), "n": int(len(x)),
            "weighted": weights is not None}


def fit_hybrid(dr, home_goals, away_goals, Z=None, covariate_names=None,
               weights=None, x0=None) -> dict:
    """Combined Groll-style covariate + Dixon-Coles hybrid (the README headline
    model): covariate-augmented symmetric log-means feeding the DC tau term, all
    parameters (a, b, beta_1..k, rho) estimated jointly by MLE.

        log lh = a + b*x + Z@beta ,  log la = a - b*x - Z@beta
        L = prod tau(h,a; lh,la,rho) * Pois(h;lh) * Pois(a;la)

    `Z` is the (n, k) covariate-differential design matrix (None -> reduces to
    plain Dixon-Coles fit). Returns a params dict carrying beta and rho, usable
    by both covariates.expected_goals (for the means) and scoreline_grid."""
    x = np.asarray(dr, dtype=float) / 400.0
    hg = np.asarray(home_goals, dtype=float)
    ag = np.asarray(away_goals, dtype=float)
    Z = np.zeros((len(x), 0)) if Z is None else np.asarray(Z, dtype=float)
    k = Z.shape[1]
    names = covariate_names or []
    w = np.ones_like(hg) if weights is None else np.asarray(weights, dtype=float)
    a0 = (x0 or {}).get("a", 0.2)
    b0 = (x0 or {}).get("b", 0.5)

    def nll(params):
        a, b = params[0], params[1]
        beta = params[2:2 + k]
        rho = params[2 + k]
        zc = Z @ beta if k else 0.0
        lh = np.exp(a + b * x + zc)
        la = np.exp(a - b * x - zc)
        t = tau(hg, ag, lh, la, rho)
        if np.any(t <= 0):
            return 1e12
        ll = hg * np.log(lh) - lh + ag * np.log(la) - la + np.log(t)
        return -float(np.sum(w * ll))

    init = np.concatenate([[a0, b0], np.zeros(k), [0.05]])
    res = minimize(nll, x0=init, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10000})
    a, b = float(res.x[0]), float(res.x[1])
    beta = {names[j]: float(res.x[2 + j]) for j in range(k)}
    return {"a": a, "b": b, "beta": beta, "covariates": names,
            "rho": float(res.x[2 + k]), "nll": float(res.fun), "n": int(len(x))}


def outcome_probs(dr: float, params: dict, max_goals: int = MAX_GOALS) -> tuple[float, float, float]:
    """P(home win, draw, away win) over 90 min from the DC-corrected joint grid.

    Returned in (home/dr-advantaged-side win, draw, loss) order to match
    goals.outcome_probs. `params` must carry a, b, rho."""
    lh, la = expected_goals(dr, params)
    grid = scoreline_grid(lh, la, params["rho"], max_goals)
    i = np.arange(max_goals + 1)
    hh, aa = np.meshgrid(i, i, indexing="ij")
    p_home = float(grid[hh > aa].sum())
    p_draw = float(np.trace(grid))
    p_away = float(grid[hh < aa].sum())
    return p_home, p_draw, p_away


def sample_scores(lh: float, la: float, rho: float, rng, size: int,
                  max_goals: int = MAX_GOALS) -> np.ndarray:
    """Draw `size` (home, away) scorelines from the DC-corrected joint grid.
    Returns int array of shape (size, 2)."""
    grid = scoreline_grid(lh, la, rho, max_goals)
    flat = grid.ravel()
    idx = rng.choice(flat.size, size=size, p=flat)
    return np.stack([idx // (max_goals + 1), idx % (max_goals + 1)], axis=1)


def time_decay_weights(dates, as_of, half_life_years: float) -> np.ndarray:
    """Exponential decay w = 0.5 ** (age_years / half_life). `dates` and
    `as_of` are ISO date strings/arrays. half_life_years<=0 -> uniform weights."""
    d = np.asarray(dates, dtype="datetime64[D]")
    ref = np.datetime64(as_of)
    age_years = (ref - d) / np.timedelta64(365, "D")
    if half_life_years is None or half_life_years <= 0:
        return np.ones(len(d))
    return 0.5 ** (age_years / half_life_years)
