"""Tests for bookmaker-consensus / overround removal (wcpred.odds)."""

import numpy as np
import pytest

from wcpred import odds


# ---------- implied_probs ----------

def test_implied_probs_sum_to_one():
    p = odds.implied_probs([2.0, 4.0, 4.0])
    assert p.sum() == pytest.approx(1.0)


def test_fair_odds_recover_true_probs():
    # [2, 4, 4] is a perfectly fair (no-margin) market: 1/2 + 1/4 + 1/4 = 1.
    # Overround removal should leave the implied probabilities essentially intact.
    p = odds.implied_probs([2.0, 4.0, 4.0])
    assert p[0] == pytest.approx(0.5, abs=1e-6)
    assert p[1] == pytest.approx(0.25, abs=1e-6)
    assert p[2] == pytest.approx(0.25, abs=1e-6)


def test_overround_removed_preserves_ordering():
    # A market with a built-in margin (raw probs sum > 1). Favourite = lowest odds.
    p = odds.implied_probs([1.5, 4.0, 7.0])
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[1] > p[2]  # ordering preserved: shortest odds -> highest prob


def test_shorter_odds_more_probable():
    p = odds.implied_probs([1.2, 10.0])
    assert p[0] > p[1]


# ---------- consensus ----------

def test_consensus_identical_rows_returns_that_row():
    row = np.array([0.5, 0.3, 0.2])
    out = odds.consensus([row, row, row])
    np.testing.assert_allclose(out, row, atol=1e-9)


def test_consensus_sums_to_one():
    out = odds.consensus([np.array([0.6, 0.3, 0.1]), np.array([0.4, 0.4, 0.2])])
    assert out.sum() == pytest.approx(1.0)


def test_consensus_geometric_mean_hand_value():
    # Two books, one outcome each: geometric mean then renormalize.
    r1 = np.array([0.5, 0.5])
    r2 = np.array([0.2, 0.8])
    g = np.array([np.sqrt(0.5 * 0.2), np.sqrt(0.5 * 0.8)])
    expected = g / g.sum()
    np.testing.assert_allclose(odds.consensus([r1, r2]), expected, atol=1e-9)


def test_consensus_between_inputs():
    # Pooled estimate must lie between the two books for each outcome.
    r1 = np.array([0.7, 0.2, 0.1])
    r2 = np.array([0.3, 0.4, 0.3])
    out = odds.consensus([r1, r2])
    for i in range(3):
        assert min(r1[i], r2[i]) <= out[i] <= max(r1[i], r2[i])


# ---------- event_h2h_consensus ----------

def _event(home, away, books):
    """books: list of {team: decimal_odds} dicts -> The Odds API event shape."""
    return {
        "home_team": home, "away_team": away,
        "bookmakers": [
            {"markets": [{"key": "h2h", "outcomes": [
                {"name": n, "price": pr} for n, pr in bk.items()]}]}
            for bk in books
        ],
    }


def test_event_h2h_consensus_basic():
    ev = _event("Spain", "Brazil",
                [{"Spain": 2.0, "Draw": 4.0, "Brazil": 4.0}])
    out = odds.event_h2h_consensus(ev)
    assert out["home"] == "Spain" and out["away"] == "Brazil"
    assert out["n_books"] == 1
    assert out["p_home"] + out["p_draw"] + out["p_away"] == pytest.approx(1.0)
    assert out["p_home"] == pytest.approx(0.5, abs=1e-6)


def test_event_h2h_consensus_counts_multiple_books():
    ev = _event("A", "B", [
        {"A": 2.0, "Draw": 4.0, "B": 4.0},
        {"A": 1.8, "Draw": 3.6, "B": 5.0},
    ])
    out = odds.event_h2h_consensus(ev)
    assert out["n_books"] == 2


def test_event_h2h_consensus_none_when_market_incomplete():
    # Missing the Draw outcome -> no usable row -> None.
    ev = _event("A", "B", [{"A": 2.0, "B": 2.0}])
    assert odds.event_h2h_consensus(ev) is None


def test_event_h2h_consensus_ignores_non_h2h_markets():
    ev = {
        "home_team": "A", "away_team": "B",
        "bookmakers": [{"markets": [
            {"key": "totals", "outcomes": [{"name": "Over", "price": 1.9}]},
        ]}],
    }
    assert odds.event_h2h_consensus(ev) is None


# ---------- outright_consensus ----------

def test_outright_consensus_restricts_to_universe_and_normalizes():
    event = {"bookmakers": [
        {"markets": [{"key": "outrights", "outcomes": [
            {"name": "Brazil", "price": 5.0},
            {"name": "France", "price": 6.0},
            {"name": "Tuvalu", "price": 1000.0},  # outside universe, dropped
        ]}]},
    ]}
    probs, n_books = odds.outright_consensus(event, universe={"Brazil", "France"})
    assert n_books == 1
    assert set(probs) == {"Brazil", "France"}
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["Brazil"] > probs["France"]  # shorter odds -> higher prob


def test_outright_consensus_skips_books_missing_universe_team():
    event = {"bookmakers": [
        {"markets": [{"key": "outrights", "outcomes": [
            {"name": "Brazil", "price": 5.0},
            {"name": "France", "price": 6.0},
        ]}]},
        {"markets": [{"key": "outrights", "outcomes": [
            {"name": "Brazil", "price": 4.0},  # missing France -> book skipped
        ]}]},
    ]}
    _, n_books = odds.outright_consensus(event, universe={"Brazil", "France"})
    assert n_books == 1


def test_outright_consensus_mid_tournament_eliminated_teams():
    # Mid-tournament: universe has 3 teams but one (Argentina) has been eliminated
    # and bookmakers no longer quote it.  The function should shrink the effective
    # universe to {Brazil, France} and return a valid consensus rather than crashing.
    event = {"bookmakers": [
        {"markets": [{"key": "outrights", "outcomes": [
            {"name": "Brazil", "price": 4.0},
            {"name": "France", "price": 5.0},
            # Argentina deliberately absent — eliminated
        ]}]},
    ]}
    probs, n_books = odds.outright_consensus(
        event, universe={"Brazil", "France", "Argentina"}
    )
    assert n_books == 1
    assert set(probs) == {"Brazil", "France"}  # Argentina dropped (not quoted)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["Brazil"] > probs["France"]


def test_outright_consensus_no_bookmakers_returns_empty():
    # No bookmaker data at all (e.g. market not yet open).
    event = {"bookmakers": []}
    probs, n_books = odds.outright_consensus(event, universe={"Brazil", "France"})
    assert probs == {}
    assert n_books == 0


def test_outright_consensus_no_book_covers_any_universe_team():
    # All quoted teams are outside the universe.
    event = {"bookmakers": [
        {"markets": [{"key": "outrights", "outcomes": [
            {"name": "Tuvalu", "price": 4.0},
        ]}]},
    ]}
    probs, n_books = odds.outright_consensus(event, universe={"Brazil", "France"})
    assert probs == {}
    assert n_books == 0
