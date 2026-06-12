"""Tests for forecast metrics (wcpred.metrics). Outcomes ordered (home, draw, away)."""

import math

import numpy as np
import pytest

from wcpred import metrics


# ---------- log_loss ----------

def test_log_loss_perfect_prediction_is_zero():
    probs = np.array([1.0, 0.0, 0.0])
    assert metrics.log_loss(probs, 0) == pytest.approx(0.0)


def test_log_loss_hand_value():
    probs = np.array([0.5, 0.3, 0.2])
    assert metrics.log_loss(probs, 0) == pytest.approx(-math.log(0.5))
    assert metrics.log_loss(probs, 1) == pytest.approx(-math.log(0.3))
    assert metrics.log_loss(probs, 2) == pytest.approx(-math.log(0.2))


def test_log_loss_zero_prob_uses_eps_no_inf():
    probs = np.array([1.0, 0.0, 0.0])
    val = metrics.log_loss(probs, 1)  # predicted 0 for the realised outcome
    assert math.isfinite(val)
    assert val == pytest.approx(-math.log(metrics.EPS))


# ---------- brier ----------

def test_brier_perfect_is_zero():
    assert metrics.brier(np.array([1.0, 0.0, 0.0]), 0) == pytest.approx(0.0)


def test_brier_hand_value():
    probs = np.array([0.5, 0.3, 0.2])
    # outcome 0 -> target (1,0,0): (0.5-1)^2 + 0.3^2 + 0.2^2
    expected = 0.25 + 0.09 + 0.04
    assert metrics.brier(probs, 0) == pytest.approx(expected)


# ---------- rps ----------

def test_rps_perfect_is_zero():
    assert metrics.rps(np.array([1.0, 0.0, 0.0]), 0) == pytest.approx(0.0)


def test_rps_worst_case_is_one():
    # Predict certain away (index 2) but home (index 0) happens -> max RPS == 1.
    probs = np.array([0.0, 0.0, 1.0])
    assert metrics.rps(probs, 0) == pytest.approx(1.0)


def test_rps_respects_ordering():
    # A near-miss (predict draw, away happens) must score lower than a far-miss
    # (predict home, away happens) -- RPS rewards being "ordinally close".
    probs_near = np.array([0.0, 1.0, 0.0])  # all mass on draw
    probs_far = np.array([1.0, 0.0, 0.0])   # all mass on home
    outcome = 2  # away
    assert metrics.rps(probs_near, outcome) < metrics.rps(probs_far, outcome)


def test_rps_hand_value():
    # probs (0.5, 0.3, 0.2), outcome home (0).
    # cp = (0.5, 0.8, 1.0); co for outcome 0 = (1, 1, 1).
    # sum over first 2 terms: (0.5-1)^2 + (0.8-1)^2 = 0.25 + 0.04 = 0.29; /2 = 0.145
    probs = np.array([0.5, 0.3, 0.2])
    assert metrics.rps(probs, 0) == pytest.approx(0.145)
