"""Reliability + isotonic calibration for 3-way forecasts."""
import numpy as np

from wcpred import calibration as cal


def test_reliability_curve_perfect_when_freq_matches_pred():
    # 1000 matches, all forecast home 60% / draw 20% / away 20%, realised exactly so.
    probs = np.tile([0.6, 0.2, 0.2], (1000, 1))
    outcomes = np.array([0] * 600 + [1] * 200 + [2] * 200)
    curve = cal.reliability_curve(probs, outcomes)
    for b in curve:
        assert abs(b["mean_pred"] - b["obs_freq"]) < 1e-9   # on the diagonal
    assert sum(b["count"] for b in curve) == 3000           # 3 classes x 1000


def test_ece_zero_when_calibrated_positive_when_not():
    probs = np.tile([0.6, 0.2, 0.2], (1000, 1))
    good = np.array([0] * 600 + [1] * 200 + [2] * 200)
    assert cal.expected_calibration_error(probs, good) < 1e-9
    # overconfident: says 90% home but home only happens 60% of the time
    over = np.tile([0.9, 0.05, 0.05], (1000, 1))
    bad = np.array([0] * 600 + [1] * 200 + [2] * 200)
    assert cal.expected_calibration_error(over, bad) > 0.1


def test_isotonic_reduces_ece_and_keeps_simplex():
    rng = np.random.default_rng(0)
    n = 3000
    ph = rng.uniform(0.4, 0.95, n)            # predicted home prob (overconfident)
    true_home = 0.2 + 0.5 * ph                # actual rate is shrunk toward 0.5
    home = rng.random(n) < true_home
    probs = np.column_stack([ph, (1 - ph) / 2, (1 - ph) / 2])
    outcomes = np.where(home, 0, rng.integers(1, 3, n))

    before = cal.expected_calibration_error(probs, outcomes)
    models = cal.fit_isotonic(probs, outcomes)
    calibrated = cal.apply_isotonic(models, probs)
    after = cal.expected_calibration_error(calibrated, outcomes)

    assert after < before                                   # better calibrated
    assert np.allclose(calibrated.sum(axis=1), 1.0)         # still a distribution
    assert (calibrated >= 0).all()


def test_temperature_softens_overconfidence_and_keeps_simplex():
    rng = np.random.default_rng(1)
    n = 3000
    ph = rng.uniform(0.5, 0.95, n)            # confident home calls...
    home = rng.random(n) < (0.25 + 0.5 * ph)  # ...that are over-confident
    probs = np.column_stack([ph, (1 - ph) / 2, (1 - ph) / 2])
    outcomes = np.where(home, 0, rng.integers(1, 3, n))

    t = cal.fit_temperature(probs, outcomes)
    assert t > 1.0                            # softens an over-confident model
    out = cal.apply_temperature(t, probs)
    assert np.allclose(out.sum(axis=1), 1.0) and (out >= 0).all()
    # softening pulls extreme probabilities toward the centre
    assert out[:, 0].max() < probs[:, 0].max()


def test_temperature_one_is_identity():
    probs = np.array([[0.6, 0.25, 0.15], [0.2, 0.3, 0.5]])
    assert np.allclose(cal.apply_temperature(1.0, probs), probs)


def test_apply_isotonic_dead_row_falls_back_to_raw():
    class _Zero:
        def transform(self, x):
            return np.zeros_like(np.asarray(x, dtype=float))

    out = cal.apply_isotonic([_Zero(), _Zero(), _Zero()], np.array([[0.5, 0.3, 0.2]]))
    assert np.allclose(out, [[0.5, 0.3, 0.2]])              # renormalised raw row
