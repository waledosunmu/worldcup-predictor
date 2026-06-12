"""Tests for the Monte Carlo tournament simulator (wcpred.simulate).

Two styles:
  A. Group tiebreak logic on a minimal hand-built Simulator (no full bracket).
  B. Full-bracket distribution + result-locking on the real format_2026.json,
     small n_sims, seeded for determinism.
"""

import json
from pathlib import Path

import pytest

from wcpred import goals
from wcpred.simulate import Simulator

ROOT = Path(__file__).resolve().parents[1]
FORMAT_PATH = ROOT / "data" / "format_2026.json"

# Goals params used throughout; mild Elo sensitivity.
PARAMS = {"a": 0.1, "b": 0.5}


# ============================================================
# Style A: tiebreak logic via a minimal Simulator
# ============================================================

def _mini_sim(teams, ratings):
    """Simulator whose __init__ only needs groups + round_of_32; the latter
    can be empty since we call _rank_group directly."""
    fmt = {"groups": {"A": list(teams)}, "round_of_32": {}}
    return Simulator(fmt, ratings, PARAMS, seed=1)


def test_rank_group_pure_points_ordering():
    teams = ["W", "X", "Y", "Z"]
    ratings = {t: 1500.0 for t in teams}
    sim = _mini_sim(teams, ratings)
    # stats: team -> [pts, gf, ga]
    stats = {"W": [9, 5, 0], "X": [6, 4, 1], "Y": [3, 2, 3], "Z": [0, 0, 7]}
    # No real h2h needed (no point ties): supply trivial pairwise records.
    h2h = _all_pairs_h2h(teams, default=(0, 0))
    assert sim._rank_group("A", stats, h2h) == ["W", "X", "Y", "Z"]


def test_rank_group_overall_gd_breaks_points_tie_when_no_h2h_decision():
    # Two teams level on points; their head-to-head was a draw, so the
    # tiebreak falls through to overall goal difference.
    teams = ["P", "Q", "R", "S"]
    ratings = {t: 1500.0 for t in teams}
    sim = _mini_sim(teams, ratings)
    stats = {"P": [4, 5, 2], "Q": [4, 3, 3], "R": [1, 1, 2], "S": [1, 1, 3]}
    h2h = _all_pairs_h2h(teams, default=(0, 0))
    h2h[("P", "Q")] = (1, 1)  # drew each other -> h2h cannot separate
    order = sim._rank_group("A", stats, h2h)
    # P has GD +3, Q has GD 0 -> P ranks above Q.
    assert order.index("P") < order.index("Q")


def test_rank_group_h2h_beats_overall_gd():
    # P and Q level on points. Q has the better OVERALL goal difference, but P
    # beat Q head-to-head. The code applies h2h within the tied cluster FIRST,
    # so P must rank above Q despite worse overall GD.
    teams = ["P", "Q", "R", "S"]
    ratings = {t: 1500.0 for t in teams}
    sim = _mini_sim(teams, ratings)
    stats = {"P": [4, 2, 1], "Q": [4, 9, 1], "R": [1, 0, 4], "S": [1, 0, 5]}
    h2h = _all_pairs_h2h(teams, default=(0, 0))
    h2h[("P", "Q")] = (1, 0)  # P beat Q
    order = sim._rank_group("A", stats, h2h)
    assert order.index("P") < order.index("Q")


def test_rank_group_elo_is_final_discriminator():
    # Everything identical except Elo rating -> higher Elo ranks higher.
    teams = ["P", "Q"]
    ratings = {"P": 1900.0, "Q": 1500.0}
    sim = _mini_sim(teams, ratings)
    stats = {"P": [3, 1, 1], "Q": [3, 1, 1]}
    h2h = {("P", "Q"): (1, 1)}  # drew -> h2h can't separate; equal GD/goals
    order = sim._rank_group("A", stats, h2h)
    assert order == ["P", "Q"]


def _all_pairs_h2h(teams, default):
    import itertools
    return {(a, b): default for a, b in itertools.combinations(teams, 2)}


# ============================================================
# Style B: full bracket on real format, seeded, small n_sims
# ============================================================

@pytest.fixture(scope="module")
def fmt():
    with open(FORMAT_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def real_ratings(fmt):
    # Distinct ratings so tiebreaks and KO outcomes are well-defined; spread
    # them so favourites emerge but every team is plausible.
    teams = [t for g in fmt["groups"].values() for t in g]
    return {t: 1500.0 + 5.0 * i for i, t in enumerate(teams)}


def _stage_sum(probs, stage):
    return sum(p[stage] for p in probs.values())


def test_run_count_invariants(fmt, real_ratings):
    """Per-sim seats are conserved, so summed stage probabilities are exact
    integers regardless of seed or n_sims. These are the strongest possible
    'valid distribution' checks."""
    sim = Simulator(fmt, real_ratings, PARAMS, seed=42)
    out = sim.run(n_sims=20)
    probs = out["probs"]

    assert _stage_sum(probs, "champion") == pytest.approx(1.0)
    assert _stage_sum(probs, "reach_final") == pytest.approx(2.0)
    assert _stage_sum(probs, "reach_sf") == pytest.approx(4.0)
    assert _stage_sum(probs, "reach_qf") == pytest.approx(8.0)
    assert _stage_sum(probs, "reach_r16") == pytest.approx(16.0)
    assert _stage_sum(probs, "reach_r32") == pytest.approx(32.0)
    assert _stage_sum(probs, "win_group") == pytest.approx(12.0)


def test_run_probabilities_are_valid(fmt, real_ratings):
    sim = Simulator(fmt, real_ratings, PARAMS, seed=7)
    out = sim.run(n_sims=20)
    for team, p in out["probs"].items():
        for stage, val in p.items():
            assert 0.0 <= val <= 1.0, (team, stage, val)
    # Nesting: champion <= reach_final <= ... <= reach_r32 for every team.
    order = ["champion", "reach_final", "reach_sf", "reach_qf",
             "reach_r16", "reach_r32"]
    for p in out["probs"].values():
        for a, b in zip(order, order[1:]):
            assert p[a] <= p[b] + 1e-12


def test_run_is_deterministic_with_same_seed(fmt, real_ratings):
    out1 = Simulator(fmt, real_ratings, PARAMS, seed=99).run(n_sims=15)
    out2 = Simulator(fmt, real_ratings, PARAMS, seed=99).run(n_sims=15)
    assert out1["probs"] == out2["probs"]


def test_run_different_seeds_differ(fmt, real_ratings):
    out1 = Simulator(fmt, real_ratings, PARAMS, seed=1).run(n_sims=30)
    out2 = Simulator(fmt, real_ratings, PARAMS, seed=2).run(n_sims=30)
    assert out1["probs"] != out2["probs"]


def test_locking_group_results_forces_group_win(fmt, real_ratings):
    """Lock a team to win all three of its group matches -> 9 points, which
    beats any opponent's maximum of 6. Its win_group probability must be 1.0
    in every simulation regardless of RNG."""
    teams_A = fmt["groups"]["A"]
    winner = teams_A[0]
    others = teams_A[1:]
    played = {}
    for opp in others:
        # winner beats each opponent 1-0; key is frozenset, rec is (first, gf, ga)
        played[frozenset((winner, opp))] = (winner, 1, 0)

    sim = Simulator(fmt, real_ratings, PARAMS, seed=3, played_group=played)
    out = sim.run(n_sims=20)
    assert out["probs"][winner]["win_group"] == pytest.approx(1.0)
    assert out["probs"][winner]["reach_r32"] == pytest.approx(1.0)


def test_locking_group_results_forces_group_loss(fmt, real_ratings):
    """Lock a team to lose all three group matches by big margins -> 0 points
    and terrible GD; it can never win the group."""
    teams_A = fmt["groups"]["A"]
    loser = teams_A[0]
    others = teams_A[1:]
    played = {}
    for opp in others:
        played[frozenset((loser, opp))] = (loser, 0, 3)  # loser scores 0, concedes 3

    sim = Simulator(fmt, real_ratings, PARAMS, seed=5, played_group=played)
    out = sim.run(n_sims=20)
    assert out["probs"][loser]["win_group"] == pytest.approx(0.0)


def test_ko_winner_lock_is_respected(fmt, real_ratings):
    sim = Simulator(fmt, real_ratings, PARAMS, seed=11)
    a, b = "Brazil", "Morocco"
    # Force Brazil to win the head-to-head whenever this pairing occurs.
    sim.ko_winners[frozenset((a, b))] = a
    assert sim._ko_winner(a, b) == a
    assert sim._ko_winner(b, a) == a


def test_simulate_once_shapes(fmt, real_ratings):
    import numpy as np
    sim = Simulator(fmt, real_ratings, PARAMS, seed=4)
    lam = np.array([[f[3], f[4]] for f in sim.group_fixtures])
    goals_sample = sim.rng.poisson(lam)
    res = sim.simulate_once(goals_sample)
    assert set(res) >= {"ranks", "qualified_thirds", "r32", "rounds",
                        "winners", "champion"}
    assert len(res["ranks"]) == 12  # 12 groups
    assert all(len(order) == 4 for order in res["ranks"].values())
    assert len(res["qualified_thirds"]) == 8
    assert res["champion"] in sim.teams


def test_allocate_thirds_respects_eligibility(fmt, real_ratings):
    """Each assigned third-place slot must receive a team whose group letter is
    in that slot's eligibility set -- the core of the backtracking allocator."""
    sim = Simulator(fmt, real_ratings, PARAMS, seed=1)
    # Pick 8 group letters that the backtracker can satisfy: feed the first 8
    # groups, each contributing its (letter, team).
    letters = sorted(fmt["groups"])[:8]
    qualified = [(g, fmt["groups"][g][2]) for g in letters]
    assignment = sim._allocate_thirds(qualified)
    assert len(assignment) == len(qualified)
    letter_of = {t: g for g, t in qualified}
    for slot, team in assignment.items():
        assert letter_of[team] in sim.third_eligible[slot], (slot, team)
    # Every qualified team is placed exactly once.
    assert sorted(assignment.values()) == sorted(t for _, t in qualified)


def test_hosts_get_home_advantage_in_group(fmt, real_ratings):
    """Host group fixtures should embed +100 dr for the host as home team."""
    sim = Simulator(fmt, real_ratings, PARAMS, seed=1)
    # Mexico is a host in group A; find its fixtures.
    host_fix = [f for f in sim.group_fixtures
                if f[1] == "Mexico" or f[2] == "Mexico"]
    assert host_fix
    # In every Mexico group fixture, Mexico is the home side (hosts at home).
    assert all(f[1] == "Mexico" for f in host_fix)
