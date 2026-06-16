"""Golden Boot (top scorer) model from martj42 goalscorers.csv (CC0).

We only have SCORER data (who scored, when) — not full line-ups — so a true
goals-per-appearance rate is not computable. Instead each player gets a SHARE of
their national team's goals over a recent point-in-time window, and we distribute
the team's model-expected tournament goals among its scorers by that share. A
player's goals already scored in 2026 are a fixed starting tally; remaining goals
are Poisson with mean (share x expected future team goals). Own goals are excluded;
penalties count.

Honest limitations (documented, not hidden):
  - No squad lists are freely/cleanly available, so the contender pool is "players
    who have scored for a 2026 finalist in the recent window" — an approximation of
    the actual 23–26-man squads, not the official rosters.
  - Goal share is a proxy for individual scoring rate; it cannot see minutes,
    penalty-taker status, or injuries.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

LOOKBACK_DAYS = 730       # ~2 years of recent form for goal shares
SEASON_START = "2026-06-01"


def _scored(goalscorers: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    """Goals in [lo, hi) excluding own goals (penalties kept)."""
    return goalscorers[(goalscorers.date >= lo) & (goalscorers.date < hi)
                       & (~goalscorers.own_goal.astype(bool))]


def recent_goal_shares(goalscorers: pd.DataFrame, as_of: str,
                       lookback_days: int = LOOKBACK_DAYS) -> dict[str, dict[str, float]]:
    """{team: {scorer: fraction of the team's goals}} over the lookback window."""
    lo = (dt.date.fromisoformat(as_of) - dt.timedelta(days=lookback_days)).isoformat()
    g = _scored(goalscorers, lo, as_of)
    shares: dict[str, dict[str, float]] = {}
    for team, grp in g.groupby("team"):
        counts = grp.scorer.value_counts()
        total = float(counts.sum())
        shares[team] = {p: c / total for p, c in counts.items()}
    return shares


def current_goals(goalscorers: pd.DataFrame, as_of: str,
                  season_start: str = SEASON_START) -> dict[str, int]:
    """{scorer: goals} already scored in the 2026 finals so far."""
    g = _scored(goalscorers, season_start, as_of)
    return {p: int(c) for p, c in g.scorer.value_counts().items()}


def simulate(contenders: list[dict], rng, n_sims: int = 50_000) -> list[dict]:
    """Per-player Golden Boot probability and expected total goals.

    contenders: [{scorer, team, current, exp_future}]. Each player's final goals =
    fixed `current` + Poisson(`exp_future`); the Golden Boot is the max each trial
    (ties shared equally). Returns the same list enriched with p_win + exp_goals,
    sorted by p_win then exp_goals.
    """
    if not contenders:
        return []
    cur = np.array([c["current"] for c in contenders], dtype=float)
    lam = np.clip([c["exp_future"] for c in contenders], 0.0, None)
    draws = rng.poisson(lam, size=(n_sims, len(contenders))) + cur
    is_top = draws == draws.max(axis=1, keepdims=True)
    p_win = (is_top / is_top.sum(axis=1, keepdims=True)).sum(axis=0) / n_sims
    out = [{"scorer": c["scorer"], "team": c["team"], "current": int(cur[i]),
            "exp_goals": round(float(cur[i] + lam[i]), 2), "p_win": float(p_win[i])}
           for i, c in enumerate(contenders)]
    out.sort(key=lambda r: (-r["p_win"], -r["exp_goals"]))
    return out
