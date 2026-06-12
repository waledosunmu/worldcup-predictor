"""Tests for the point-in-time Elo engine (wcpred.elo)."""

import math

import pandas as pd
import pytest

from wcpred import elo


# ---------- expectancy formula ----------

def test_expectancy_zero_diff_is_half():
    assert elo.expectancy(0.0) == pytest.approx(0.5)


def test_expectancy_monotonic_and_bounded():
    lo, mid, hi = elo.expectancy(-400), elo.expectancy(0), elo.expectancy(400)
    assert 0.0 < lo < mid < hi < 1.0
    # 400-point edge -> ~10:1 expected result by the standard Elo curve
    assert hi == pytest.approx(10.0 / 11.0, abs=1e-9)


def test_expectancy_symmetry():
    # We(dr) + We(-dr) == 1 for any dr
    for dr in (50, 137.5, 400, -220):
        assert elo.expectancy(dr) + elo.expectancy(-dr) == pytest.approx(1.0)


# ---------- K-factor ----------

def test_k_factor_world_cup():
    assert elo.k_factor("FIFA World Cup") == 60.0


def test_k_factor_continental_final():
    assert elo.k_factor("UEFA Euro") == 50.0
    assert elo.k_factor("Copa América") == 50.0


def test_k_factor_qualification_substring_match():
    # "qualification" substring -> 40, regardless of the confederation prefix
    assert elo.k_factor("FIFA World Cup qualification") == 40.0
    assert elo.k_factor("UEFA Euro qualification") == 40.0
    assert elo.k_factor("UEFA Nations League") == 40.0


def test_k_factor_friendly():
    assert elo.k_factor("Friendly") == 20.0


def test_k_factor_default_other_tournament():
    assert elo.k_factor("Some Random Cup") == 30.0


# ---------- margin-of-victory multiplier ----------

def test_mov_multiplier_table():
    assert elo.mov_multiplier(0) == 1.0
    assert elo.mov_multiplier(1) == 1.0
    assert elo.mov_multiplier(-1) == 1.0  # uses abs()
    assert elo.mov_multiplier(2) == 1.5
    assert elo.mov_multiplier(3) == 1.75
    # N >= 4 : 1.75 + (N-3)/8
    assert elo.mov_multiplier(4) == pytest.approx(1.75 + 1 / 8)
    assert elo.mov_multiplier(5) == pytest.approx(1.75 + 2 / 8)
    assert elo.mov_multiplier(-4) == pytest.approx(1.75 + 1 / 8)  # symmetric


# ---------- single-update mechanics ----------

def _one_match_frame(home, away, hs, as_, tourn, neutral, date="2020-01-01"):
    return pd.DataFrame([{
        "date": date, "home_team": home, "away_team": away,
        "home_score": hs, "away_score": as_,
        "tournament": tourn, "neutral": neutral,
    }])


def test_zero_sum_symmetry_of_updates():
    # Both teams start at BASE; the home gain must equal the away loss exactly.
    df = _one_match_frame("A", "B", 1, 0, "Friendly", neutral=True)
    r = elo.compute_ratings(df)
    gain = r["A"] - elo.BASE_RATING
    loss = elo.BASE_RATING - r["B"]
    assert gain == pytest.approx(loss)
    # Total rating is conserved.
    assert r["A"] + r["B"] == pytest.approx(2 * elo.BASE_RATING)
    assert gain > 0  # the winner gained


def test_draw_with_equal_ratings_no_change():
    df = _one_match_frame("A", "B", 1, 1, "Friendly", neutral=True)
    r = elo.compute_ratings(df)
    assert r["A"] == pytest.approx(elo.BASE_RATING)
    assert r["B"] == pytest.approx(elo.BASE_RATING)


def test_home_advantage_applied_when_not_neutral():
    # Non-neutral: home gets +100 dr, so an upset away win moves ratings more
    # than the same scoreline at a neutral venue.
    neutral = elo.compute_ratings(_one_match_frame("A", "B", 0, 1, "Friendly", True))
    home = elo.compute_ratings(_one_match_frame("A", "B", 0, 1, "Friendly", False))
    # B (away) won; with home advantage the expectancy for A was higher,
    # so B's gain is larger in the non-neutral case.
    assert (home["B"] - elo.BASE_RATING) > (neutral["B"] - elo.BASE_RATING)


def test_mov_amplifies_delta():
    small = elo.compute_ratings(_one_match_frame("A", "B", 1, 0, "Friendly", True))
    big = elo.compute_ratings(_one_match_frame("A", "B", 4, 0, "Friendly", True))
    assert (big["A"] - elo.BASE_RATING) > (small["A"] - elo.BASE_RATING)


def test_known_delta_value():
    # Equal ratings, neutral, World Cup, 1-0: we=0.5, w=1, k=60, mov=1
    # delta = 60 * (1 - 0.5) = 30
    df = _one_match_frame("A", "B", 1, 0, "FIFA World Cup", neutral=True)
    r = elo.compute_ratings(df)
    assert r["A"] - elo.BASE_RATING == pytest.approx(30.0)
    assert elo.BASE_RATING - r["B"] == pytest.approx(30.0)


# ---------- point-in-time correctness (no future leakage) ----------

def test_as_of_is_exclusive_excludes_match_on_boundary():
    df = _one_match_frame("A", "B", 5, 0, "Friendly", True, date="2020-06-01")
    # as_of equal to the match date -> match excluded (strict < comparison)
    r = elo.compute_ratings(df, as_of="2020-06-01")
    assert r == {}  # no matches processed, no ratings


def test_future_match_does_not_leak():
    df = pd.DataFrame([
        {"date": "2019-01-01", "home_team": "A", "away_team": "B",
         "home_score": 1, "away_score": 0, "tournament": "Friendly", "neutral": True},
        {"date": "2021-01-01", "home_team": "A", "away_team": "B",
         "home_score": 9, "away_score": 0, "tournament": "Friendly", "neutral": True},
    ])
    past_only = elo.compute_ratings(df, as_of="2020-01-01")
    full = elo.compute_ratings(df)
    # The 2021 thrashing must not influence the 2020 snapshot.
    assert past_only["A"] != pytest.approx(full["A"])
    # Past-only equals running just the first match.
    one = elo.compute_ratings(df[df.date < "2020-01-01"])
    assert past_only["A"] == pytest.approx(one["A"])


def test_drops_unplayed_matches():
    df = _one_match_frame("A", "B", None, None, "Friendly", True)
    r = elo.compute_ratings(df)
    assert r == {}


# ---------- rating_history_frame: pre-match discipline ----------

def test_rating_history_frame_pre_match_ratings():
    df = pd.DataFrame([
        {"date": "2019-01-01", "home_team": "A", "away_team": "B",
         "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": True},
        {"date": "2019-06-01", "home_team": "A", "away_team": "B",
         "home_score": 0, "away_score": 0, "tournament": "FIFA World Cup", "neutral": True},
    ])
    hist = elo.rating_history_frame(df)
    # First match: both pre-match ratings are BASE, dr == 0.
    assert hist.iloc[0]["elo_home_pre"] == pytest.approx(elo.BASE_RATING)
    assert hist.iloc[0]["elo_away_pre"] == pytest.approx(elo.BASE_RATING)
    assert hist.iloc[0]["dr"] == pytest.approx(0.0)
    # Second match pre-rating for A reflects the +30 from match one (WC 1-0).
    assert hist.iloc[1]["elo_home_pre"] == pytest.approx(elo.BASE_RATING + 30.0)
    assert hist.iloc[1]["dr"] == pytest.approx(60.0)  # 1530 - 1470, neutral
