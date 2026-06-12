"""Bookmaker-consensus model (Leitner/Zeileis/Hornik 2010 recipe).

Per bookmaker: remove the overround by assuming a constant margin delta across
outcomes — solve delta so that implied probabilities sum to one, where
p_i = 1 / (1 + (raw_i - 1) * delta) for decimal odds raw_i. Then form the
consensus across bookmakers as the renormalized geometric mean (equivalent to
averaging log-probabilities, close to the paper's log-odds averaging).
"""

import numpy as np


def implied_probs(decimal_odds: list[float]) -> np.ndarray:
    """Overround-free probabilities for one bookmaker's market."""
    raw = np.asarray(decimal_odds, dtype=float)

    def total(delta: float) -> float:
        return float(np.sum(1.0 / (1.0 + (raw - 1.0) * delta)))

    lo, hi = 0.2, 100.0
    if total(lo) < 1.0:  # extreme underround; fall back to proportional
        p = 1.0 / raw
        return p / p.sum()
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if total(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    p = 1.0 / (1.0 + (raw - 1.0) * (lo + hi) / 2.0)
    return p / p.sum()


def consensus(prob_rows: list[np.ndarray]) -> np.ndarray:
    """Renormalized geometric mean across bookmakers (rows align on outcomes)."""
    logp = np.log(np.clip(np.vstack(prob_rows), 1e-12, 1.0))
    g = np.exp(logp.mean(axis=0))
    return g / g.sum()


def event_h2h_consensus(event: dict) -> dict | None:
    """The Odds API event -> {p_home, p_draw, p_away, n_books}."""
    home, away = event["home_team"], event["away_team"]
    rows = []
    for bk in event.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] != "h2h":
                continue
            prices = {o["name"]: o["price"] for o in mkt["outcomes"]}
            if home in prices and away in prices and "Draw" in prices:
                rows.append(implied_probs([prices[home], prices["Draw"], prices[away]]))
    if not rows:
        return None
    p = consensus(rows)
    return {"home": home, "away": away, "p_home": float(p[0]),
            "p_draw": float(p[1]), "p_away": float(p[2]), "n_books": len(rows)}


def outright_consensus(event: dict, canon=lambda n: n,
                       universe: set | None = None) -> tuple[dict, int]:
    """The Odds API outrights event -> ({team: probability}, n_books).

    Restricts to `universe` (e.g. the 48 qualified teams) — books may quote
    stale non-qualified teams — and uses the books that cover every universe
    team. Probabilities are overround-corrected per book, then renormalized
    over the universe.
    """
    books = []
    for bk in event.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] == "outrights":
                books.append({canon(o["name"]): o["price"] for o in mkt["outcomes"]})
    teams = sorted(universe if universe is not None
                   else set().union(*[set(b) for b in books]))
    rows = [implied_probs([prices[t] for t in teams])
            for prices in books if set(teams) <= set(prices)]
    p = consensus(rows)
    return dict(zip(teams, map(float, p))), len(rows)


def snapshot_to_consensus(winner_snap: dict, match_snap: dict,
                          canon=lambda n: n, universe: set | None = None) -> dict:
    """Derive market consensus from a raw odds snapshot pair (winner + match h2h).

    Shared by run_consensus.py, the history backfill, and the odds-movement tracker.
    Returns {captured, n_outright_books, outright_consensus, matches} with each match
    `{home, away, commence, p_home, p_draw, p_away, n_books}` (canonical team names,
    restricted to `universe` when given).
    """
    outright, n_books = outright_consensus(winner_snap["data"][0], canon=canon,
                                           universe=universe)
    matches = []
    for ev in match_snap["data"]:
        c = event_h2h_consensus(ev)
        if c is None:
            continue
        c["home"], c["away"] = canon(c["home"]), canon(c["away"])
        c["commence"] = ev["commence_time"]
        if universe is None or (c["home"] in universe and c["away"] in universe):
            matches.append(c)
    return {"captured": winner_snap.get("captured"), "n_outright_books": n_books,
            "outright_consensus": outright, "matches": matches}
