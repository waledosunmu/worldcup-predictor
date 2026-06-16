"""Live 2026 World Cup results: fetch finished matches and overlay their scores
onto the committed martj42 results.csv (and shootouts.csv).

The committed snapshot ships every 2026 World Cup fixture as a schedule stub with
NA scores; martj42 fills them in as matches are played, but with a lag. This
module provides a faster live overlay (football-data.org) AND treats martj42 as
the automatic fallback: it only ever writes a score onto a row that is currently
NA, never overwriting an existing result, and it keys matches by the unordered
team pair within the tournament window — robust to home/away orientation flips
between the live source and the snapshot.

Team-name canonicalization is the critical join step: the live source uses its
own spellings ("USA", "Korea Republic"); we map them onto the 48 martj42 squad
names from format_2026.json and FAIL LOUDLY on any World-Cup team we cannot map,
so a silent name mismatch can never quietly reproduce the "0 matches played"
bug this module exists to fix.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

WC_TOURNAMENT = "FIFA World Cup"
WINDOW_START = "2026-06-01"

# football-data.org (and other feeds) spelling -> martj42 / format_2026 name.
# Only entries that actually differ from the martj42 spelling are listed; an
# exact-match name passes through untouched.
LIVE_NAME_MAP = {
    "USA": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "Czechia": "Czech Republic",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "DR Congo (Congo-Kinshasa)": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Iran": "Iran",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Curacao": "Curaçao",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Saudi Arabia": "Saudi Arabia",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}


def squad_names(fmt: dict) -> set[str]:
    """The 48 martj42 team names for the qualified squads."""
    return {t for g in fmt["groups"].values() for t in g}


def canonicalize(name: str, valid: set[str]) -> str:
    """Map a live-feed team name to its martj42 spelling.

    Raises KeyError if the name is neither a known squad nor in LIVE_NAME_MAP —
    callers convert this into a loud, non-silent failure.
    """
    if name in valid:
        return name
    mapped = LIVE_NAME_MAP.get(name)
    if mapped is not None and mapped in valid:
        return mapped
    raise KeyError(name)


def load_env(root: Path) -> None:
    """Populate os.environ from .env, matching scripts/snapshot_odds.py."""
    env = root / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def fetch_football_data(api_key: str, competition: str = "WC") -> list[dict]:
    """Finished 2026 World Cup matches from football-data.org v4.

    Returns a list of normalized dicts:
        {home, away, home_score, away_score, pen_home, pen_away, date}
    (team names still in the feed's own spelling — canonicalize() later).
    """
    import json
    import urllib.request

    url = (f"https://api.football-data.org/v4/competitions/{competition}/matches"
           "?status=FINISHED")
    req = urllib.request.Request(url, headers={"X-Auth-Token": api_key})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)

    out = []
    for m in data.get("matches", []):
        score = m.get("score", {})
        ft = score.get("fullTime", {})
        if ft.get("home") is None or ft.get("away") is None:
            continue
        # The v4 docs describe `penalties` both as an aggregate {home,away} object
        # and (elsewhere) as a per-kick list [{team, scored, ...}]. Handle both,
        # and never raise: a parse miss must not silently kill the whole overlay
        # (it is wrapped in a broad except upstream). Unknown shape -> no shootout.
        pens = score.get("penalties")
        pen_home = pen_away = None
        if isinstance(pens, dict):
            pen_home, pen_away = pens.get("home"), pens.get("away")
        elif isinstance(pens, list):
            ph = sum(1 for k in pens if k.get("team") == m["homeTeam"]["name"] and k.get("scored"))
            pa = sum(1 for k in pens if k.get("team") == m["awayTeam"]["name"] and k.get("scored"))
            if ph or pa:
                pen_home, pen_away = ph, pa
        out.append({
            "home": m["homeTeam"]["name"],
            "away": m["awayTeam"]["name"],
            "home_score": int(ft["home"]),
            "away_score": int(ft["away"]),
            "pen_home": pen_home,
            "pen_away": pen_away,
            "date": (m.get("utcDate") or "")[:10] or None,
        })
    return out


def merge_results(results: pd.DataFrame, shootouts: pd.DataFrame,
                  matches: list[dict], valid: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Overlay finished `matches` onto the WC2026 rows of `results`.

    Only NA scores are filled (never overwriting an existing result), keyed by
    the unordered team pair within the WC window — orientation-robust. A 0-0
    knockout (a draw that went to penalties) appends/updates a shootouts.csv row
    using the live penalty count. Returns (results, shootouts, summary).

    Raises KeyError (via canonicalize) on any unmappable team name so a silent
    join miss is impossible.
    """
    results = results.copy()
    shootouts = shootouts.copy()

    is_wc = ((results.tournament == WC_TOURNAMENT)
             & (results.date >= WINDOW_START))
    # unordered-pair -> list of row indices for WC fixtures
    pair_rows: dict[frozenset, list[int]] = {}
    for idx in results.index[is_wc]:
        key = frozenset((results.at[idx, "home_team"], results.at[idx, "away_team"]))
        pair_rows.setdefault(key, []).append(idx)

    summary = {"matched": 0, "filled": 0, "already_had_score": 0,
               "no_fixture": 0, "shootouts_added": 0, "unmapped": []}

    for m in matches:
        try:
            home = canonicalize(m["home"], valid)
            away = canonicalize(m["away"], valid)
        except KeyError as e:
            summary["unmapped"].append(str(e))
            continue
        key = frozenset((home, away))
        rows = pair_rows.get(key)
        if not rows:
            summary["no_fixture"] += 1
            continue
        summary["matched"] += 1
        # pick the (first) row that still needs a score
        target = None
        for idx in rows:
            if pd.isna(results.at[idx, "home_score"]):
                target = idx
                break
        if target is None:
            summary["already_had_score"] += 1
            continue

        # orient the live score to the snapshot row's home/away
        row_home = results.at[target, "home_team"]
        hs, as_ = m["home_score"], m["away_score"]
        ph, pa = m.get("pen_home"), m.get("pen_away")
        if row_home != home:  # feed had teams the other way round
            hs, as_ = as_, hs
            ph, pa = pa, ph
        results.at[target, "home_score"] = hs
        results.at[target, "away_score"] = as_
        summary["filled"] += 1

        # knockout draw decided on penalties -> record the shootout winner
        if hs == as_ and ph is not None and pa is not None and ph != pa:
            date = results.at[target, "date"]
            r_home = results.at[target, "home_team"]
            r_away = results.at[target, "away_team"]
            winner = r_home if ph > pa else r_away
            existing = shootouts[(shootouts.date >= WINDOW_START)
                                 & (shootouts.home_team == r_home)
                                 & (shootouts.away_team == r_away)]
            if existing.empty:
                shootouts = pd.concat([shootouts, pd.DataFrame([{
                    "date": date, "home_team": r_home, "away_team": r_away,
                    "winner": winner, "first_shooter": winner,
                }])], ignore_index=True)
                summary["shootouts_added"] += 1

    return results, shootouts, summary


def write_results_csv(results: pd.DataFrame, path) -> None:
    """Write results.csv in the martj42 snapshot's canonical format.

    The 72 unplayed WC2026 rows carry NA scores, which forces the score columns
    to float and `neutral` to bool on read; a naive ``to_csv`` then rewrites all
    ~49k rows (every ``0`` becomes ``0.0``, every ``FALSE`` becomes ``False``),
    producing a multi-MB diff each run. Casting scores back to nullable ints
    (NA -> empty cell) and re-uppercasing the booleans keeps the on-disk format
    byte-identical to the curl'd martj42 source, so only the rows whose scores
    actually changed show up in the diff.
    """
    out = results.copy()
    for col in ("home_score", "away_score"):
        out[col] = out[col].astype("Int64")
    if out["neutral"].dtype == bool:
        out["neutral"] = out["neutral"].map({True: "TRUE", False: "FALSE"})
    out.to_csv(path, index=False, na_rep="NA")
