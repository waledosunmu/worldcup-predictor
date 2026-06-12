"""Odds-movement observability: how much does the market move before kickoff?

The kickoff-aware cadence concentrates paid snapshots in the hours before kickoff. This
module measures whether that buys anything — if the consensus barely moves late, the
cadence can be relaxed. `build_record` makes the compact per-snapshot series row
(written by run_consensus.py to odds_timeseries.jsonl); `analyze` turns the accumulated
series into drift metrics + a plain-English verdict. Both pure / unit-tested; the
markdown report lives in scripts/analyze_odds_movement.py.
"""
from __future__ import annotations

import datetime as dt
from statistics import mean, median

FINAL_WINDOW_H = 3.0


def build_record(captured: str, n_books: int, outright: dict, matches: list[dict]) -> dict:
    """Compact time-series row from a derived consensus (drops model/ensemble fields)."""
    return {
        "captured": captured,
        "n_outright_books": n_books,
        "outright": dict(outright),
        "matches": [{"home": m["home"], "away": m["away"], "commence": m.get("commence"),
                     "p_home": m["p_home"], "p_draw": m["p_draw"], "p_away": m["p_away"],
                     "n_books": m.get("n_books")} for m in matches],
    }


def _parse(ts: str) -> dt.datetime:
    ts = ts.replace("Z", "")
    for fmt in ("%Y-%m-%dT%H%M%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(ts, fmt)
        except ValueError:
            pass
    return dt.datetime.fromisoformat(ts)


def _key(m: dict) -> str:
    return f"{m['home']}|{m['away']}"


def analyze(records: list[dict]) -> dict:
    """Aggregate pre-kickoff drift of the match consensus across captures.

    For each match we follow the favourite outcome's consensus probability across all
    captures taken before kickoff, and report how far it drifts overall vs. only in the
    final FINAL_WINDOW_H hours.
    """
    # gather captures per match: list of (captured_dt, commence_dt, [ph,pd,pa])
    per_match: dict[str, list[tuple]] = {}
    for r in records:
        cap = _parse(r["captured"])
        for m in r.get("matches", []):
            if not m.get("commence"):
                continue
            ko = _parse(m["commence"])
            if cap > ko:
                continue  # post-kickoff capture — ignore
            per_match.setdefault(_key(m), []).append(
                (cap, ko, [m["p_home"], m["p_draw"], m["p_away"]], m))

    rows = []
    for key, caps in per_match.items():
        caps.sort(key=lambda t: t[0])
        if len(caps) < 2:
            continue
        ko = caps[-1][1]
        fav = max(range(3), key=lambda i: caps[-1][2][i])  # favourite at last capture
        series = [((ko - cap).total_seconds() / 3600, probs[fav]) for cap, _, probs, _ in caps]
        open_p, close_p = series[0][1], series[-1][1]
        total = abs(close_p - open_p)
        # Final-window move is only defined if we actually captured inside the window;
        # measure drift from the ~FINAL_WINDOW_H mark to the close. None otherwise, so we
        # never imply "no late movement" when we simply have no late captures.
        inside = [p for h, p in series if h < FINAL_WINDOW_H]
        early = [p for h, p in series if h >= FINAL_WINDOW_H]
        final_move = abs(close_p - (early[-1] if early else open_p)) if inside else None
        rows.append({
            "home": caps[-1][3]["home"], "away": caps[-1][3]["away"],
            "commence": caps[-1][3]["commence"], "n_captures": len(caps),
            "favourite": ["home", "draw", "away"][fav],
            "open_prob": round(open_p, 4), "close_prob": round(close_p, 4),
            "total_drift_pp": round(total * 100, 2),
            "final_window_move_pp": None if final_move is None else round(final_move * 100, 2),
        })

    # outright market swing: per team, max-min champion prob across captures
    out_swing = 0.0
    seen_teams = set().union(*[set(r.get("outright", {})) for r in records]) if records else set()
    for t in seen_teams:
        vals = [r["outright"][t] for r in records if t in r.get("outright", {})]
        if len(vals) >= 2:
            out_swing = max(out_swing, max(vals) - min(vals))

    if not rows:
        return {"n_records": len(records), "n_matches_tracked": 0,
                "outright_max_swing_pp": round(out_swing * 100, 2),
                "verdict": "insufficient data — need ≥2 pre-kickoff captures of a match"}

    totals = [r["total_drift_pp"] for r in rows]
    finals = [r["final_window_move_pp"] for r in rows if r["final_window_move_pp"] is not None]
    if finals:
        mfinal = mean(finals)
        verdict = (
            f"avg favourite-prob move in the final {FINAL_WINDOW_H:g}h = {mfinal:.2f}pp. "
            + ("Minimal late movement — snapshot cadence can likely be reduced." if mfinal < 1.0
               else "Moderate late movement — current cadence is reasonable." if mfinal < 3.0
               else "Substantial late movement — frequent near-kickoff snapshots are worthwhile."))
        final_summary = {"mean": round(mfinal, 2), "max": round(max(finals), 2),
                         "n_matches": len(finals)}
    else:
        verdict = (f"No captures within the final {FINAL_WINDOW_H:g}h of kickoff yet — the "
                   f"late-movement signal accrues once the kickoff-aware cadence runs near "
                   f"kickoffs. Mean total pre-kickoff drift so far = {mean(totals):.2f}pp.")
        final_summary = {"mean": None, "max": None, "n_matches": 0}
    return {
        "n_records": len(records),
        "n_matches_tracked": len(rows),
        "mean_captures_per_match": round(mean(r["n_captures"] for r in rows), 2),
        "total_drift_pp": {"mean": round(mean(totals), 2),
                           "median": round(median(totals), 2), "max": round(max(totals), 2)},
        "final_window_move_pp": final_summary,
        "outright_max_swing_pp": round(out_swing * 100, 2),
        "verdict": verdict,
        "per_match": sorted(rows, key=lambda r: -r["total_drift_pp"]),
    }
