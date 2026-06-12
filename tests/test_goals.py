"""Tests for the Elo-diff -> expected-goals Poisson model (wcpred.goals)."""

import math

import numpy as np
import pytest

from wcpred import goals

# A representative parameter set (shape only matters: a is the baseline log-rate,
# b the Elo sensitivity). Values are in the realistic fitted range.
PARAMS = {"a": 0.2, "b": 0.5}


# ---------- expected_goals ----------

def test_equal_strength_gives_equal_expected_goals():
    lh, la = goals.expected_goals(0.0, PARAMS)
    assert lh == pytest.approx(la)
    assert lh == pytest.approx(math.exp(PARAMS["a"]))


def test_expected_goals_monotonic_in_dr():
    # Stronger side (higher dr) -> more expected goals for it, fewer for opponent.
    lh_lo, la_lo = goals.expected_goals(0.0, PARAMS)
    lh_hi, la_hi = goals.expected_goals(300.0, PARAMS)
    assert lh_hi > lh_lo  # favourite scores more
    assert la_hi < la_lo  # underdog scores fewer


def test_expected_goals_symmetry():
    # expected_goals(dr) reversed == expected_goals(-dr)
    lh, la = goals.expected_goals(250.0, PARAMS)
    lh_neg, la_neg = goals.expected_goals(-250.0, PARAMS)
    assert lh == pytest.approx(la_neg)
    assert la == pytest.approx(lh_neg)


def test_expected_goals_positive():
    lh, la = goals.expected_goals(-500.0, PARAMS)
    assert lh > 0 and la > 0  # exp() is always positive


# ---------- outcome_probs ----------

def test_outcome_probs_sum_to_one():
    p_win, p_draw, p_loss = goals.outcome_probs(120.0, PARAMS)
    assert p_win + p_draw + p_loss == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in (p_win, p_draw, p_loss))


def test_outcome_probs_symmetric_at_zero():
    p_win, p_draw, p_loss = goals.outcome_probs(0.0, PARAMS)
    # Equal strength -> win prob equals loss prob.
    assert p_win == pytest.approx(p_loss)
    assert p_draw > 0.0


def test_outcome_probs_favourite_more_likely_to_win():
    p_win, _, p_loss = goals.outcome_probs(300.0, PARAMS)
    assert p_win > p_loss


# ---------- fit ----------

def test_fit_recovers_planted_parameters():
    # Generate Poisson data from known (a, b) and check MLE recovery.
    rng = np.random.default_rng(0)
    true_a, true_b = 0.25, 0.6
    n = 4000
    dr = rng.uniform(-400, 400, size=n)
    x = dr / 400.0
    lh = np.exp(true_a + true_b * x)
    la = np.exp(true_a - true_b * x)
    home_goals = rng.poisson(lh)
    away_goals = rng.poisson(la)

    out = goals.fit(dr, home_goals, away_goals)
    assert out["n"] == n
    assert out["a"] == pytest.approx(true_a, abs=0.05)
    assert out["b"] == pytest.approx(true_b, abs=0.08)


def test_fit_returns_expected_keys():
    rng = np.random.default_rng(1)
    dr = rng.uniform(-200, 200, size=200)
    out = goals.fit(dr, rng.poisson(1.3, 200), rng.poisson(1.1, 200))
    assert set(out) == {"a", "b", "nll", "n"}
    assert isinstance(out["a"], float)
