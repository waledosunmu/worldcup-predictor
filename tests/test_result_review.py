"""Post-match pick + review helpers in explain.py."""
from wcpred.explain import pick_from, result_review

M_HOME = {"p_home": 0.7, "p_draw": 0.2, "p_away": 0.1}   # favours home
M_AWAY = {"p_home": 0.2, "p_draw": 0.2, "p_away": 0.6}   # favours away


def test_pick_from_returns_modal_side():
    assert pick_from(M_HOME) == ("home", 0.7)
    assert pick_from(M_AWAY) == ("away", 0.6)
    assert pick_from({"p_home": 0.2, "p_draw": 0.5, "p_away": 0.3})[0] == "draw"


def test_review_model_hit_with_score_and_winner():
    r = result_review("Mexico", "South Africa", 2, 0, "home", M_HOME)
    assert "Mexico 2–0 South Africa — Mexico won." in r
    assert "called it" in r and "70%" in r


def test_review_model_miss_reports_actual_prob():
    r = result_review("Haiti", "Scotland", 0, 1, "away", M_HOME)  # model liked home
    assert "Scotland won" in r
    assert "missed" in r and "10%" in r  # gave the away win only 10%


def test_review_includes_market_agreement_and_disagreement():
    agree = result_review("Mexico", "South Africa", 2, 0, "home", M_HOME, M_HOME)
    assert "market agreed" in agree and "also called it" in agree
    disagree = result_review("Mexico", "South Africa", 0, 2, "away", M_HOME, M_AWAY)
    assert "market instead leaned" in disagree and "which landed" in disagree


def test_review_draw_has_no_winner_clause():
    r = result_review("Canada", "Bosnia", 1, 1, "draw", M_HOME)
    assert "— a draw." in r and "won." not in r
