"""Tests for wcpred.pointwise — the shared point-in-time match model.

The load-bearing guarantee (CLAUDE.md cardinal rule #1) is leak-freeness: a model
built as of date D must depend ONLY on matches dated < D, so appending or deleting
any match dated >= D leaves ratings and goals params bit-for-bit unchanged. A
passing "the numbers look right" check would NOT catch a params-refit leak; this
invariance check does.
"""
import sys
from pathlib import Path

import pandas as pd

from wcpred import pointwise
from wcpred.simulate import Simulator

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "raw" / "results.csv"

AS_OF = "2026-06-29"  # a knockout match date during the 2026 tournament


def _results():
    return pd.read_csv(RESULTS)


def test_build_asof_model_ignores_matches_on_or_after_as_of():
    """Ratings + params are invariant to any row dated >= as_of (leak-free)."""
    base = _results()
    # A blatantly outcome-changing future match: it must have zero influence.
    future = pd.DataFrame([{
        "date": "2026-12-31", "home_team": "Brazil", "away_team": "Japan",
        "home_score": 0, "away_score": 9, "tournament": "FIFA World Cup",
        "neutral": True,
    }])
    polluted = pd.concat([base, future], ignore_index=True)

    clean = pointwise.build_asof_model(base, AS_OF, model="v0")
    with_future = pointwise.build_asof_model(polluted, AS_OF, model="v0")

    assert clean["ratings"] == with_future["ratings"]
    assert clean["params"] == with_future["params"]


def test_build_asof_model_is_deterministic():
    r = _results()
    a = pointwise.build_asof_model(r, AS_OF, model="v0")
    b = pointwise.build_asof_model(r, AS_OF, model="v0")
    assert a["ratings"] == b["ratings"] and a["params"] == b["params"]


def _mini_sim(ratings, params, dixon_coles=False):
    """Simulator with just enough state for matchup_probs (no bracket needed)."""
    sim = Simulator.__new__(Simulator)
    sim.ratings = ratings
    sim.params = params
    sim.dixon_coles = dixon_coles
    sim.team_form = None
    return sim


def test_matchup_probs_deterministic_and_normalised():
    sim = _mini_sim({"A": 1900.0, "B": 1700.0}, {"a": 0.1, "b": 0.5})
    p1 = sim.matchup_probs("A", "B", host_adv=0.0)
    p2 = sim.matchup_probs("A", "B", host_adv=0.0)
    assert p1 == p2  # analytic, no RNG (cardinal rule #2)
    total = p1["p_home"] + p1["p_draw"] + p1["p_away"]
    assert abs(total - 1.0) < 1e-3
    assert p1["p_home"] > p1["p_away"]  # stronger side favoured


def test_matchup_probs_host_advantage_shifts_toward_home():
    sim = _mini_sim({"A": 1800.0, "B": 1800.0}, {"a": 0.1, "b": 0.5})
    neutral = sim.matchup_probs("A", "B", host_adv=0.0)
    hosted = sim.matchup_probs("A", "B", host_adv=100.0)
    assert hosted["p_home"] > neutral["p_home"]
