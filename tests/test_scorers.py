"""Golden Boot goal-share model."""
import numpy as np
import pandas as pd

from wcpred import scorers


def _gs(rows):
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team", "team",
                                       "scorer", "minute", "own_goal", "penalty"])


GOALS = _gs([
    ["2025-09-01", "Argentina", "Chile", "Argentina", "Messi", 10, False, False],
    ["2025-09-01", "Argentina", "Chile", "Argentina", "Messi", 20, False, True],   # pen counts
    ["2025-09-01", "Argentina", "Chile", "Argentina", "Lautaro", 30, False, False],
    ["2025-09-01", "Argentina", "Chile", "Chile", "OwnGuy", 40, True, False],       # own goal: ignored
    ["2020-01-01", "Argentina", "Peru", "Argentina", "Old", 5, False, False],       # before window
    ["2026-06-11", "Argentina", "X", "Argentina", "Lautaro", 50, False, False],     # 2026 tally
])


def test_recent_goal_shares_window_and_own_goals():
    shares = scorers.recent_goal_shares(GOALS, as_of="2026-06-15", lookback_days=730)
    arg = shares["Argentina"]
    # window goals: Messi 2, Lautaro 2 (incl 2026-06-11); Old excluded (too old); own goal excluded
    assert abs(arg["Messi"] - 0.5) < 1e-9 and abs(arg["Lautaro"] - 0.5) < 1e-9
    assert "Old" not in arg
    assert abs(sum(arg.values()) - 1.0) < 1e-9      # shares sum to 1


def test_current_goals_counts_2026_only_excluding_own():
    cur = scorers.current_goals(GOALS, as_of="2026-06-15")
    assert cur == {"Lautaro": 1}                    # only the 2026-06-11 goal


def test_simulate_probabilities_sum_to_one_and_favour_the_strong():
    rng = np.random.default_rng(0)
    contenders = [
        {"scorer": "Star", "team": "A", "current": 1, "exp_future": 5.0},
        {"scorer": "Mid", "team": "B", "current": 0, "exp_future": 2.0},
        {"scorer": "Weak", "team": "C", "current": 0, "exp_future": 0.5},
    ]
    out = scorers.simulate(contenders, rng, n_sims=20000)
    assert abs(sum(p["p_win"] for p in out) - 1.0) < 1e-9   # someone always wins
    assert out[0]["scorer"] == "Star"                        # highest expected -> most likely
    assert out[0]["p_win"] > out[1]["p_win"] > out[2]["p_win"]
    assert out[0]["exp_goals"] == 6.0                        # current 1 + future 5


def test_simulate_is_deterministic_and_handles_empty():
    c = [{"scorer": "P", "team": "A", "current": 0, "exp_future": 3.0}]
    a = scorers.simulate(c, np.random.default_rng(3), n_sims=5000)
    b = scorers.simulate(c, np.random.default_rng(3), n_sims=5000)
    assert a == b
    assert scorers.simulate([], np.random.default_rng(1)) == []
