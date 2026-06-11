"""Probabilistic forecast metrics for ordered W/D/L outcomes."""

import numpy as np

EPS = 1e-12


def log_loss(probs: np.ndarray, outcome: int) -> float:
    return -float(np.log(max(probs[outcome], EPS)))


def brier(probs: np.ndarray, outcome: int) -> float:
    o = np.zeros(3)
    o[outcome] = 1.0
    return float(np.sum((probs - o) ** 2))


def rps(probs: np.ndarray, outcome: int) -> float:
    """Ranked probability score over the ordered outcomes (home, draw, away)."""
    o = np.zeros(3)
    o[outcome] = 1.0
    cp, co = np.cumsum(probs), np.cumsum(o)
    return float(np.sum((cp[:2] - co[:2]) ** 2) / 2.0)
