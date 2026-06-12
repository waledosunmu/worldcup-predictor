"""Tests for the templated narrative generator (wcpred.explain)."""

import pandas as pd
import pytest

from wcpred import explain


def _results():
    return pd.DataFrame([
        # A vs B history
        {"date": "2018-06-01", "home_team": "Aland", "away_team": "Bland",
         "home_score": 2.0, "away_score": 1.0, "tournament": "Friendly", "neutral": True},
        {"date": "2019-06-01", "home_team": "Bland", "away_team": "Aland",
         "home_score": 0.0, "away_score": 0.0, "tournament": "FIFA World Cup", "neutral": True},
        # Aland other results, for recent form
        {"date": "2020-01-01", "home_team": "Aland", "away_team": "Cland",
         "home_score": 3.0, "away_score": 1.0, "tournament": "Friendly", "neutral": True},
        {"date": "2020-02-01", "home_team": "Cland", "away_team": "Aland",
         "home_score": 2.0, "away_score": 0.0, "tournament": "Friendly", "neutral": True},
    ])


# ---------- h2h_record ----------

def test_h2h_record_counts_from_a_perspective():
    rec = explain.h2h_record(_results(), "Aland", "Bland", before="2026-01-01")
    # Match 1: Aland 2-1 win; Match 2: 0-0 draw. -> 1W 1D 0L in 2 meetings.
    assert rec["n"] == 2
    assert (rec["w"], rec["d"], rec["l"]) == (1, 1, 0)
    assert "last" in rec
    assert "2019" in rec["last"]  # most recent meeting referenced


def test_h2h_record_respects_before_cutoff():
    rec = explain.h2h_record(_results(), "Aland", "Bland", before="2019-01-01")
    # Only the 2018 match is before the cutoff.
    assert rec["n"] == 1
    assert rec["w"] == 1


def test_h2h_record_never_met():
    rec = explain.h2h_record(_results(), "Aland", "Zland", before="2026-01-01")
    assert rec["n"] == 0
    assert "last" not in rec


# ---------- recent_form ----------

def test_recent_form_string_and_goals():
    f = explain.recent_form(_results(), "Aland", before="2026-01-01", n=8)
    # Aland: 2-1 W, 0-0 D, 3-1 W, 0-2 L (chronological) -> "WDWL"
    assert f["string"] == "WDWL"
    assert f["n"] == 4
    # goals for: 2+0+3+0 = 5 ; against: 1+0+1+2 = 4
    assert f["gf"] == 5
    assert f["ga"] == 4


def test_recent_form_respects_n_limit():
    f = explain.recent_form(_results(), "Aland", before="2026-01-01", n=2)
    assert f["n"] == 2
    assert f["string"] == "WL"  # last two chronologically


# ---------- fixture_explanation ----------

def _fx():
    return {
        "home": "Aland", "away": "Bland",
        "xg_home": 1.8, "xg_away": 0.9,
        "p_home": 0.55, "p_draw": 0.25, "p_away": 0.20,
    }


def test_fixture_explanation_renders_and_references_fields():
    ratings = {"Aland": 1820.0, "Bland": 1710.0}
    ranks = {"Aland": 8, "Bland": 15}
    out = explain.fixture_explanation(_fx(), ratings, ranks, _results(), as_of="2026-01-01")
    assert isinstance(out, str) and out
    assert "Aland" in out and "Bland" in out
    assert "1820" in out  # Elo rendered
    assert "110-point edge" in out  # |1820 - 1710|
    assert "Head-to-head" in out  # h2h branch (they have met)
    assert "form" in out


def test_fixture_explanation_market_divergence_note():
    ratings = {"Aland": 1820.0, "Bland": 1710.0}
    ranks = {"Aland": 8, "Bland": 15}
    market = {"n_books": 6, "p_home": 0.40, "p_draw": 0.30, "p_away": 0.30}
    out = explain.fixture_explanation(_fx(), ratings, ranks, _results(),
                                      as_of="2026-01-01", market=market)
    assert "Bookmaker consensus" in out
    # model p_home 0.55 vs market 0.40 -> +0.15 >= DIVERGENCE_PP -> "more confident"
    assert "more confident" in out


def test_fixture_explanation_never_met_branch():
    ratings = {"Aland": 1820.0, "Zland": 1500.0}
    ranks = {"Aland": 8, "Zland": 40}
    fx = dict(_fx(), away="Zland")
    out = explain.fixture_explanation(fx, ratings, ranks, _results(), as_of="2026-01-01")
    assert "never met" in out


# ---------- team_blurb ----------

def test_team_blurb_renders_simulation_fields():
    adv = {"reach_r32": 0.85, "win_group": 0.45, "reach_sf": 0.18, "champion": 0.06}
    out = explain.team_blurb("Aland", adv, rating=1820.0, rank=8, group="A")
    assert "Group A" in out
    assert "85%" in out  # reach_r32 formatted as percent
    assert "6.0%" in out  # champion formatted to 1 decimal


def test_team_blurb_market_divergence_note():
    adv = {"reach_r32": 0.85, "win_group": 0.45, "reach_sf": 0.18, "champion": 0.10}
    out = explain.team_blurb("Aland", adv, rating=1820.0, rank=8, group="A",
                             market_champ=0.04)
    # 0.10 - 0.04 = 0.06 >= 0.03 -> divergence note present
    assert "higher on" in out
    assert "Aland" in out
