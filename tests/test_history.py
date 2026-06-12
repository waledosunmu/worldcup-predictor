"""Probability + match-prediction history builders."""
from wcpred import history


def _latest(as_of="2026-06-12", played=False):
    fx = {"group": "A", "home": "Mexico", "away": "South Africa"}
    if played:
        fx.update(played=True, score_home=2, score_away=0)
    else:
        fx.update(played=False, p_home=0.7, p_draw=0.2, p_away=0.1)
    return {
        "as_of": as_of, "n_played_group": 1 if played else 0, "n_played_ko": 0,
        "ratings": {"Mexico": 1980.0, "South Africa": 1663.0},
        "advancement": {
            "Mexico": {"champion": 0.05, "reach_final": 0.2},
            "South Africa": {"champion": 0.005, "reach_final": 0.03},
        },
        "group_fixtures": [fx],
    }


def _consensus():
    return {
        "n_outright_books": 5,
        "outright_consensus": {"Mexico": 0.04, "South Africa": 0.004},
        "matches": [{"home": "Mexico", "away": "South Africa", "p_home": 0.6,
                     "p_draw": 0.25, "p_away": 0.15, "n_books": 40,
                     "commence": "2026-06-11T19:00:00Z"}],
    }


def test_build_history_record_pulls_model_and_market():
    rec = history.build_history_record(_latest(), _consensus(), "2026-06-12T12:00:00Z")
    assert rec["as_of"] == "2026-06-12"
    assert rec["model"]["champion"]["Mexico"] == 0.05
    assert rec["model"]["reach_final"]["Mexico"] == 0.2
    assert rec["model"]["elo"]["Mexico"] == 1980.0
    assert rec["market"]["champion"]["Mexico"] == 0.04
    assert rec["n_outright_books"] == 5


def test_upsert_history_dedups_same_as_of_and_sorts():
    a = history.build_history_record(_latest("2026-06-11"), _consensus(), "t1")
    b = history.build_history_record(_latest("2026-06-12"), _consensus(), "t2")
    changed = _latest("2026-06-12")
    changed["advancement"]["Mexico"]["champion"] = 0.09  # genuinely new data
    b2 = history.build_history_record(changed, _consensus(), "t3")
    recs = history.upsert_history(history.upsert_history([a], b), b2)
    assert [r["as_of"] for r in recs] == ["2026-06-11", "2026-06-12"]  # deduped, sorted
    assert recs[-1]["timestamp"] == "t3"  # changed same-day data -> later write wins


def test_build_match_predictions_joins_market_skips_played():
    preds = history.build_match_predictions(_latest(played=False), _consensus(), "t")
    assert len(preds) == 1
    p = preds[0]
    assert p["key"] == "Mexico|South Africa"
    assert p["model"] == {"p_home": 0.7, "p_draw": 0.2, "p_away": 0.1}
    assert p["market"]["n_books"] == 40 and p["market"]["p_home"] == 0.6
    # a played fixture is not a prediction candidate
    assert history.build_match_predictions(_latest(played=True), _consensus(), "t") == []


def test_predictions_refresh_while_unplayed_then_freeze_on_result():
    # first lock at opening-ish probs
    p1 = history.build_match_predictions(_latest(played=False), _consensus(), "t1")
    recs = history.merge_match_predictions([], p1, [])
    assert recs[0]["model"]["p_home"] == 0.7

    # still unplayed next run with newer probs -> overwrite (closing prediction)
    newer = _latest(played=False)
    newer["group_fixtures"][0].update(p_home=0.75, p_draw=0.18, p_away=0.07)
    p2 = history.build_match_predictions(newer, _consensus(), "t2")
    recs = history.merge_match_predictions(recs, p2, [])
    assert recs[0]["model"]["p_home"] == 0.75 and not recs[0]["played"]

    # now played -> attach result, freeze; a later stray prediction must NOT overwrite
    played = [_latest(played=True)["group_fixtures"][0] | {"home": "Mexico", "away": "South Africa"}]
    recs = history.merge_match_predictions(recs, [], played)
    assert recs[0]["played"] and recs[0]["result"]["outcome"] == "home"
    assert recs[0]["model"]["p_home"] == 0.75  # closing prediction retained
    recs = history.merge_match_predictions(recs, p1, [])  # stray re-lock attempt
    assert recs[0]["model"]["p_home"] == 0.75  # frozen, unchanged


def test_played_without_prediction_records_null_model():
    recs = history.merge_match_predictions([], [], [_latest(played=True)["group_fixtures"][0]
                                                     | {"home": "Mexico", "away": "South Africa"}])
    assert recs[0]["played"] and recs[0]["model"] is None  # excluded from scoring


def test_upsert_history_idempotent_ignoring_timestamp():
    a = history.build_history_record(_latest("2026-06-12"), _consensus(), "t1")
    again = history.build_history_record(_latest("2026-06-12"), _consensus(), "t2")
    recs = history.upsert_history([a], again)
    assert len(recs) == 1 and recs[0]["timestamp"] == "t1"  # unchanged data kept


def test_predictions_idempotent_when_probs_unchanged():
    p1 = history.build_match_predictions(_latest(played=False), _consensus(), "t1")
    recs = history.merge_match_predictions([], p1, [])
    p2 = history.build_match_predictions(_latest(played=False), _consensus(), "t2")
    recs2 = history.merge_match_predictions(recs, p2, [])
    assert recs2[0]["predicted_at"] == "t1"  # same probs -> no predicted_at churn


def test_jsonl_roundtrip(tmp_path):
    recs = [{"as_of": "2026-06-11", "x": 1}, {"as_of": "2026-06-12", "x": 2}]
    p = tmp_path / "h.jsonl"
    history.write_jsonl(p, recs)
    assert history.load_jsonl(p) == recs
    assert history.load_jsonl(tmp_path / "missing.jsonl") == []
