"""Probability calibration + reliability for 3-way (home/draw/away) forecasts.

Reliability asks: when the model says an outcome has probability p, does it happen
about p of the time? We measure it on the one-vs-rest pooled (predicted, occurred)
pairs across the three outcomes, and recalibrate with per-class isotonic regression
(monotone, non-parametric) renormalised back to a simplex. Calibration is learned only
from past data and applied out-of-sample — the backtest fits it on prior editions only.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

N_BINS = 10


def _onevsrest(probs: np.ndarray, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (N,3) probs + (N,) outcomes into pooled (3N,) predicted / occurred."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    pred = probs.reshape(-1)
    occurred = (outcomes[:, None] == np.arange(probs.shape[1])).astype(float).reshape(-1)
    return pred, occurred


def reliability_curve(probs: np.ndarray, outcomes: np.ndarray,
                      n_bins: int = N_BINS) -> list[dict]:
    """Binned reliability over pooled one-vs-rest pairs.

    Returns one entry per occupied bin: bin range, mean predicted probability, observed
    frequency, and the count of (forecast, outcome) pairs that landed in it.
    """
    pred, occurred = _onevsrest(probs, outcomes)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        out.append({"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                    "mean_pred": float(pred[mask].mean()),
                    "obs_freq": float(occurred[mask].mean()), "count": n})
    return out


def expected_calibration_error(probs: np.ndarray, outcomes: np.ndarray,
                               n_bins: int = N_BINS) -> float:
    """Count-weighted mean gap between predicted probability and observed frequency."""
    pred, occurred = _onevsrest(probs, outcomes)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bins - 1)
    total = pred.size
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n:
            ece += (n / total) * abs(pred[mask].mean() - occurred[mask].mean())
    return float(ece)


def _temper(logp: np.ndarray, t: float) -> np.ndarray:
    z = logp / t
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Single-parameter temperature that minimises log loss (T>1 softens over-confidence).

    Robust where per-class isotonic over-fits: it can't drive any probability to 0, so it
    never blows up log loss on a sparse, well-calibrated model.
    """
    from scipy.optimize import minimize_scalar
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    rows = np.arange(len(outcomes))

    def nll(t: float) -> float:
        p = _temper(logp, t)
        return -float(np.log(np.clip(p[rows, outcomes], 1e-12, 1.0)).mean())

    return float(minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded").x)


def apply_temperature(t: float, probs: np.ndarray) -> np.ndarray:
    """Temperature-scale a (N,3) probability matrix; rows stay on the simplex."""
    logp = np.log(np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0))
    return _temper(logp, t)


def fit_isotonic(probs: np.ndarray, outcomes: np.ndarray) -> list[IsotonicRegression]:
    """Fit one monotone isotonic calibrator per outcome class (one-vs-rest)."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    models = []
    for k in range(probs.shape[1]):
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(probs[:, k], (outcomes == k).astype(float))
        models.append(ir)
    return models


def apply_isotonic(models: list[IsotonicRegression], probs: np.ndarray) -> np.ndarray:
    """Apply per-class calibrators and renormalise each row back to a valid simplex.

    A row whose calibrated values all collapse to zero falls back to the raw row, so
    the result is always a proper probability distribution.
    """
    probs = np.asarray(probs, dtype=float)
    cal = np.column_stack([models[k].transform(probs[:, k])
                           for k in range(probs.shape[1])])
    sums = cal.sum(axis=1, keepdims=True)
    dead = sums[:, 0] <= 1e-12
    cal[dead] = probs[dead]
    sums[dead, 0] = probs[dead].sum(axis=1)
    return cal / sums
