"""Odds-movement drift analysis."""
from wcpred import odds_movement


def _rec(captured, commence, ph, pd, pa):
    return {"captured": captured, "n_outright_books": 5, "outright": {"Spain": 0.2},
            "matches": [{"home": "A", "away": "B", "commence": commence,
                         "p_home": ph, "p_draw": pd, "p_away": pa, "n_books": 30}]}


KO = "2026-06-20T18:00:00Z"


def test_build_record_keeps_only_compact_match_fields():
    rec = odds_movement.build_record("2026-06-20T120000Z", 5, {"Spain": 0.2},
        [{"home": "A", "away": "B", "commence": KO, "p_home": 0.5, "p_draw": 0.3,
          "p_away": 0.2, "n_books": 30, "model": {"p_home": 0.6}, "ensemble": {}}])
    m = rec["matches"][0]
    assert set(m) == {"home", "away", "commence", "p_home", "p_draw", "p_away", "n_books"}
    assert "model" not in m  # dropped for compactness


def test_insufficient_data_single_capture():
    a = odds_movement.analyze([_rec("2026-06-20T060000Z", KO, 0.5, 0.3, 0.2)])
    assert a["n_matches_tracked"] == 0 and "insufficient" in a["verdict"]


def test_total_drift_and_final_window_move():
    # 12h out then 1h out (inside 3h); favourite (home) drifts 0.50 -> 0.56
    recs = [_rec("2026-06-20T060000Z", KO, 0.50, 0.30, 0.20),
            _rec("2026-06-20T170000Z", KO, 0.56, 0.27, 0.17)]
    a = odds_movement.analyze(recs)
    assert a["n_matches_tracked"] == 1
    r = a["per_match"][0]
    assert r["favourite"] == "home"
    assert abs(r["total_drift_pp"] - 6.0) < 1e-6
    assert abs(r["final_window_move_pp"] - 6.0) < 1e-6   # 0.50 (3h mark) -> 0.56
    assert a["final_window_move_pp"]["n_matches"] == 1
    assert "final 3h" in a["verdict"]


def test_no_final_window_when_all_captures_early():
    # both captures > 3h before kickoff -> final-window undefined, honest verdict
    recs = [_rec("2026-06-20T060000Z", KO, 0.50, 0.30, 0.20),
            _rec("2026-06-20T120000Z", KO, 0.55, 0.28, 0.17)]
    a = odds_movement.analyze(recs)
    assert a["per_match"][0]["final_window_move_pp"] is None
    assert a["final_window_move_pp"]["mean"] is None
    assert "No captures within the final" in a["verdict"]


def test_post_kickoff_captures_ignored():
    recs = [_rec("2026-06-20T060000Z", KO, 0.50, 0.30, 0.20),
            _rec("2026-06-20T200000Z", KO, 0.9, 0.05, 0.05)]  # after kickoff -> dropped
    a = odds_movement.analyze(recs)
    assert a["n_matches_tracked"] == 0  # only one valid pre-kickoff capture remains
