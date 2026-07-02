"""Probability + prediction time-series for the trend graph and live track-record.

`forecast_latest.json` is overwritten every run, so on its own there is nothing to
plot a trend from. These helpers persist two compact, append-only series:

- `probability_history.jsonl` — one record per `as_of` date (same-day reruns
  overwrite), capturing per-team model champion / reach-final / Elo and the market
  champion consensus. Drives the championship trend graph and the movers widget.
- `match_predictions.jsonl` — one record per fixture. Its model + market W/D/L is
  refreshed every run while the match is unplayed, then FROZEN the moment it is played
  (so we keep the last pre-kickoff / "closing" prediction), and the actual result is
  attached. Drives the model-vs-market live track-record.

Pure builders/mergers live here (unit-tested); `scripts/append_history.py` is the thin
CLI that wires them to the JSON artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path


# ---------- jsonl IO ----------

def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in records))


# ---------- probability history (per as_of) ----------

def build_history_record(latest: dict, consensus: dict, timestamp: str) -> dict:
    """One probability-history point from a forecast_latest + consensus pair."""
    adv = latest["advancement"]
    return {
        "timestamp": timestamp,
        "as_of": latest["as_of"],
        "n_played_group": latest.get("n_played_group", 0),
        "n_played_ko": latest.get("n_played_ko", 0),
        "n_outright_books": consensus.get("n_outright_books", 0),
        "model": {
            "champion": {t: adv[t]["champion"] for t in adv},
            "reach_final": {t: adv[t]["reach_final"] for t in adv},
            "elo": dict(latest["ratings"]),
        },
        "market": {"champion": dict(consensus.get("outright_consensus", {}))},
    }


def _eq_ignoring(a: dict, b: dict, ignore=("timestamp",)) -> bool:
    drop = lambda d: {k: v for k, v in d.items() if k not in ignore}
    return drop(a) == drop(b)


def upsert_history(records: list[dict], new: dict, key: str = "as_of") -> list[dict]:
    """Insert `new`, overwriting any existing record with the same key, sorted by key.

    If an existing same-key record is identical apart from its `timestamp`, it is kept
    unchanged so no-op reruns don't churn the file (and produce empty commits).
    """
    by_key = {r[key]: r for r in records}
    cur = by_key.get(new[key])
    if cur is None or not _eq_ignoring(cur, new):
        by_key[new[key]] = new
    return [by_key[k] for k in sorted(by_key)]


# ---------- match predictions (per fixture, fill-once) ----------

def _match_key(home: str, away: str) -> str:
    return f"{home}|{away}"


def build_match_predictions(latest: dict, consensus: dict, predicted_at: str) -> list[dict]:
    """Pre-kickoff predictions for the currently-upcoming fixtures.

    Model W/D/L comes from forecast_latest (the live model), market W/D/L from the
    matching consensus fixture. Only unplayed fixtures that have model probabilities
    are returned — these are candidates to lock.
    """
    market_by_pair = {(m["home"], m["away"]): m for m in consensus.get("matches", [])}
    preds = []
    fixtures = latest.get("group_fixtures", []) + latest.get("knockout_fixtures", [])
    for fx in fixtures:
        if fx.get("played") or "p_home" not in fx:
            continue
        mk = market_by_pair.get((fx["home"], fx["away"]))
        preds.append({
            "key": _match_key(fx["home"], fx["away"]),
            "home": fx["home"], "away": fx["away"], "group": fx.get("group"),
            "commence": mk["commence"] if mk else None,
            "predicted_at": predicted_at,
            "model": {"p_home": fx["p_home"], "p_draw": fx["p_draw"],
                      "p_away": fx["p_away"]},
            "market": ({"p_home": mk["p_home"], "p_draw": mk["p_draw"],
                        "p_away": mk["p_away"], "n_books": mk["n_books"]}
                       if mk else None),
            "played": False,
            "result": None,
        })
    return preds


def _outcome(score_home: int, score_away: int) -> str:
    return ("home" if score_home > score_away
            else "away" if score_away > score_home else "draw")


def merge_match_predictions(existing: list[dict], new_preds: list[dict],
                            played_fixtures: list[dict]) -> list[dict]:
    """Refresh predictions while unplayed, freeze on kickoff, then attach results.

    - A still-unplayed fixture's prediction is overwritten with the latest probs, so a
      record always holds the most recent pre-kickoff ("closing") prediction.
    - A fixture already frozen (`played` True) is never overwritten.
    - When a fixture is played, its `result` is attached and it is frozen. A fixture
      played with no prior prediction is recorded with `model`/`market` null (excluded
      from track-record scoring).
    """
    by_key = {r["key"]: r for r in existing}

    for p in new_preds:
        cur = by_key.get(p["key"])
        if cur is not None and cur.get("played"):
            continue  # frozen — keep the locked pre-kickoff prediction
        if cur is not None and cur.get("model") == p["model"] \
                and cur.get("market") == p["market"]:
            continue  # unchanged probs — keep existing (no predicted_at churn)
        by_key[p["key"]] = {**p,
                            "played": bool(cur and cur.get("played")),
                            "result": cur.get("result") if cur else None}

    for fx in played_fixtures:
        if not fx.get("played"):
            continue
        key = _match_key(fx["home"], fx["away"])
        sh, sa = int(fx["score_home"]), int(fx["score_away"])
        result = {"score_home": sh, "score_away": sa, "outcome": _outcome(sh, sa)}
        rec = by_key.get(key)
        if rec is None:
            by_key[key] = {
                "key": key, "home": fx["home"], "away": fx["away"],
                "group": fx.get("group"), "commence": None, "predicted_at": None,
                "model": None, "market": None, "played": True, "result": result,
            }
        else:
            rec["played"] = True
            rec["result"] = result

    return [by_key[k] for k in sorted(by_key)]
