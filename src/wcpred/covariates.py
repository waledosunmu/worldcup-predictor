"""Groll-style covariate-augmented goals model.

Extends the Elo-difference -> Poisson model (goals.py) with additional
covariates entering the log-mean, following Groll et al. (JQAS 2019):

    log lambda_home = a + b*(dr/400) + sum_k beta_k * z_k
    log lambda_away = a - b*(dr/400) - sum_k beta_k * z_k

Covariates z_k are signed *differentials* (home minus away quantity) so the
model stays symmetric, exactly like the Elo-difference term. Only covariates
that can be computed point-in-time from the data already in this repo are
implemented; covariates requiring external data (market value, squad age,
Champions-League player counts) are declared in COVARIATE_SPEC with the exact
source needed, and are simply absent from the design matrix until that data is
supplied -- they are never fabricated.

IMPLEMENTED (derivable from data/raw/results.csv, strictly backward-looking):
  form_diff   - (home recent goal-difference per match) minus (away's), over a
                rolling window of the last N matches before the fixture date.
  host        - +1 if home team plays in its own country (non-neutral venue),
                else 0. NOTE: largely collinear with the +100 home advantage
                already baked into dr by the Elo engine; included only as an
                optional diagnostic and off by default.

NOT IMPLEMENTED -- external data required (see COVARIATE_SPEC):
  market_value_diff, squad_age_diff, cl_players_diff, confederation, ...
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Declared interface for the full Groll-style covariate set. `available=False`
# entries are scaffolded but require data this repo does not ship; populate
# `source` data and add a builder to enable them. Nothing here is invented.
COVARIATE_SPEC = {
    "form_diff": {
        "available": True,
        "differential": True,
        "desc": "rolling recent goal-difference-per-match, home minus away",
        "source": "data/raw/results.csv (computed point-in-time in build_form_frame)",
    },
    "host": {
        "available": True,
        "differential": True,
        "desc": "home team in its own country; redundant with dr home-adv",
        "source": "data/raw/results.csv neutral/country columns",
    },
    "market_value_diff": {
        "available": False,
        "differential": True,
        "desc": "log total squad market value, home minus away (Groll's top predictor)",
        "source": "transfermarkt squad valuations per tournament -- NOT in repo",
    },
    "squad_age_diff": {
        "available": False,
        "differential": True,
        "desc": "mean squad age, home minus away",
        "source": "per-tournament squad lists with birthdates -- NOT in repo",
    },
    "cl_players_diff": {
        "available": False,
        "differential": True,
        "desc": "# players at Champions-League clubs, home minus away",
        "source": "squad-to-club mapping + UCL participant list -- NOT in repo",
    },
    "confederation": {
        "available": False,
        "differential": False,
        "desc": "team confederation (non-differential; needs interaction terms)",
        "source": "team->confederation map; derivable but leakage-prone, deferred",
    },
}

IMPLEMENTED_COVARIATES = [k for k, v in COVARIATE_SPEC.items()
                          if v["available"] and v["differential"]]


def build_form_frame(hist: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add a strictly backward-looking `form_diff` column to a rating-history
    frame (from elo.rating_history_frame). For each match, each team's form is
    its mean goal difference over its previous `window` completed matches
    BEFORE that date; teams with no history get 0. `form_diff` = home - away.

    Mirrors the point-in-time discipline of rating_history_frame: a single
    forward pass, recording the pre-match value and only then updating state.
    """
    hist = hist.sort_values("date").reset_index(drop=True)
    from collections import deque, defaultdict
    recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def form(team):
        d = recent[team]
        return float(np.mean(d)) if d else 0.0

    home_form = np.empty(len(hist))
    away_form = np.empty(len(hist))
    cols = ["home_team", "away_team", "home_score", "away_score"]
    for i, (h, a, hs, as_) in enumerate(hist[cols].itertuples(index=False)):
        home_form[i] = form(h)
        away_form[i] = form(a)
        recent[h].append(hs - as_)
        recent[a].append(as_ - hs)
    out = hist.copy()
    out["home_form"] = home_form
    out["away_form"] = away_form
    out["form_diff"] = home_form - away_form
    return out


def current_form(hist: pd.DataFrame, window: int = 5,
                 as_of: str | None = None) -> dict[str, float]:
    """Each team's form *now*: mean goal difference over its last `window`
    completed matches strictly before `as_of` (point-in-time). Teams with no
    prior matches are absent (callers treat missing as 0.0).

    This is the forward-looking counterpart of build_form_frame: that records the
    pre-match form for historical rows; this exposes the final per-team state so
    the live forecast can attach a `form_diff` to as-yet-unplayed fixtures.
    """
    from collections import defaultdict, deque
    h = hist if as_of is None else hist[hist.date < as_of]
    h = h.sort_values("date")
    recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
    cols = ["home_team", "away_team", "home_score", "away_score"]
    for hh, aa, hs, as_ in h[cols].itertuples(index=False):
        recent[hh].append(hs - as_)
        recent[aa].append(as_ - hs)
    return {t: float(np.mean(d)) for t, d in recent.items() if d}


def add_host_covariate(frame: pd.DataFrame) -> pd.DataFrame:
    """`host` differential: +1 when the home team is at home (non-neutral)."""
    out = frame.copy()
    out["host"] = np.where(out["neutral"].astype(bool), 0.0, 1.0)
    return out


def _design(frame: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    """Stack the requested covariate differential columns into (n, k)."""
    if not covariates:
        return np.zeros((len(frame), 0))
    return np.column_stack([frame[c].to_numpy(dtype=float) for c in covariates])


def fit(dr, home_goals, away_goals, Z=None, covariate_names=None,
        weights=None, x0=None) -> dict:
    """Poisson-regression MLE of (a, b, beta_1..beta_k) for the symmetric
    log-mean model. `Z` is the (n, k) covariate-differential design matrix
    (None/empty -> reduces exactly to the v0 goals.fit). Returns a params dict
    consumable by expected_goals()."""
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
        beta = params[2:]
        zc = Z @ beta if k else 0.0
        lh = np.exp(a + b * x + zc)
        la = np.exp(a - b * x - zc)
        return -float(np.sum(w * (hg * np.log(lh) - lh + ag * np.log(la) - la)))

    init = np.concatenate([[a0, b0], np.zeros(k)])
    res = minimize(nll, x0=init, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 8000})
    a, b = float(res.x[0]), float(res.x[1])
    beta = {names[j]: float(res.x[2 + j]) for j in range(k)}
    return {"a": a, "b": b, "beta": beta, "covariates": names,
            "nll": float(res.fun), "n": int(len(x))}


def expected_goals(dr: float, z: dict | None, params: dict) -> tuple[float, float]:
    """Covariate-augmented expected goals. `z` maps covariate name -> the
    fixture's differential value (home minus away). Missing covariates -> 0.
    With no covariates this equals goals.expected_goals."""
    x = dr / 400.0
    beta = params.get("beta", {})
    z = z or {}
    zc = sum(beta[c] * float(z.get(c, 0.0)) for c in beta)
    return (float(np.exp(params["a"] + params["b"] * x + zc)),
            float(np.exp(params["a"] - params["b"] * x - zc)))
