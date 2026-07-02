"""One-time, idempotent backfill of pre-kickoff predictions for knockout matches
that were already played before the knockout model-vs-market track record existed.

The forward pipeline (run_forecast -> append_history) locks a prediction for each
UPCOMING knockout tie, so future rounds get the same track record as the group
stage. The knockout matches played before that wiring existed have no locked
prediction — and one cannot be manufactured from today's settled-bracket model
without leaking their results (CLAUDE.md cardinal rule #1). This script
reconstructs a legitimate CLOSING (last pre-kickoff) prediction for each:

  model  — wcpred.pointwise.build_asof_model(results, as_of=match_date): the exact
           point-in-time model the forecast would have published that morning,
           using only matches dated < match_date. This is the backtest discipline.
  market — the matchup's h2h consensus from the newest output/consensus.json commit
           whose commit time is STRICTLY BEFORE the match's kickoff (the closing
           line). If none exists, the match is left model-only (market null) — the
           honest outcome, never a post-kickoff line.

It writes into output/match_predictions.jsonl, force-replacing only records that
are missing or still model-null (so it never clobbers a genuinely locked
prediction and is safe to re-run, including after a pipeline run).

Usage: python scripts/backfill_ko_predictions.py [--dry-run]
"""
import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpred import history, pointwise
from wcpred.simulate import Simulator, load_format

PRED_PATH = ROOT / "output/match_predictions.jsonl"
FORECAST_PATH = ROOT / "output/forecast_latest.json"


def _parse_iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _outcome(sh: int, sa: int) -> str:
    return "home" if sh > sa else "away" if sa > sh else "draw"


def consensus_commits() -> list[tuple[str, dt.datetime]]:
    """(commit_sha, commit_time) for every commit touching consensus.json, newest first."""
    out = subprocess.check_output(
        ["git", "-C", str(ROOT), "log", "--format=%H %cI", "--", "output/consensus.json"],
        text=True)
    commits = []
    for line in out.splitlines():
        sha, iso = line.split()
        commits.append((sha, _parse_iso(iso)))
    return commits


def _consensus_at(sha: str) -> dict | None:
    try:
        return json.loads(subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{sha}:output/consensus.json"], text=True))
    except subprocess.CalledProcessError:
        return None


def closing_markets(matches: list[dict], commits: list[tuple[str, dt.datetime]]) -> dict:
    """For each played KO matchup, the h2h consensus from the newest commit whose
    commit time is strictly before that match's kickoff. Oriented to the match's
    (home, away). Returns {frozenset(home,away): {market, commence, predicted_at}}."""
    wanted = {frozenset((m["home"], m["away"])): m for m in matches}
    found: dict = {}
    for sha, ctime in commits:  # newest -> oldest
        if len(found) == len(wanted):
            break
        cj = _consensus_at(sha)
        if cj is None:
            continue
        by_set = {frozenset((e["home"], e["away"])): e for e in cj.get("matches", [])}
        for key, want in wanted.items():
            if key in found or key not in by_set:
                continue
            e = by_set[key]
            commence = _parse_iso(e["commence"]) if e.get("commence") else None
            if commence is None or ctime >= commence:
                continue  # not strictly pre-kickoff — never take a post-kickoff line
            # orient probs to the match's home/away
            if e["home"] == want["home"]:
                ph, pd_, pa = e["p_home"], e["p_draw"], e["p_away"]
            else:
                ph, pd_, pa = e["p_away"], e["p_draw"], e["p_home"]
            found[key] = {
                "market": {"p_home": ph, "p_draw": pd_, "p_away": pa,
                           "n_books": e.get("n_books")},
                "commence": e["commence"],
                "predicted_at": ctime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written, but don't touch the file")
    args = ap.parse_args()

    forecast = json.load(open(FORECAST_PATH))
    played = [k for k in forecast.get("knockout_fixtures", []) if k.get("played")]
    if not played:
        print("no played knockout matches in forecast_latest.json — nothing to backfill")
        return
    print(f"{len(played)} played knockout match(es) to backfill")

    results = pd.read_csv(ROOT / "data/raw/results.csv")
    fmt = load_format(ROOT / "data/format_2026.json")

    markets = closing_markets(played, consensus_commits())

    # point-in-time model, one bundle per distinct match date (cache the refit)
    bundles: dict[str, Simulator] = {}

    def sim_for(as_of: str) -> Simulator:
        if as_of not in bundles:
            b = pointwise.build_asof_model(results, as_of, model="hybrid")
            bundles[as_of] = Simulator(fmt, b["ratings"], b["params"],
                                       team_form=b["team_form"], dixon_coles=b["hybrid"])
        return bundles[as_of]

    records = []
    for k in played:
        home, away, date = k["home"], k["away"], k["date"]
        probs = sim_for(date).matchup_probs(home, away, 0.0)  # neutral knockout
        mk = markets.get(frozenset((home, away)))
        sh, sa = int(k["score_home"]), int(k["score_away"])
        rec = {
            "key": history._match_key(home, away),
            "home": home, "away": away, "group": None,
            "commence": mk["commence"] if mk else None,
            "predicted_at": mk["predicted_at"] if mk else date,
            "model": {"p_home": probs["p_home"], "p_draw": probs["p_draw"],
                      "p_away": probs["p_away"]},
            "market": mk["market"] if mk else None,
            "played": True,
            "result": {"score_home": sh, "score_away": sa, "outcome": _outcome(sh, sa)},
        }
        records.append(rec)
        tag = "model+market" if mk else "model-only (no pre-kickoff line)"
        print(f"  {date} {home} {sh}-{sa} {away}: {tag}")

    if args.dry_run:
        print("(dry run — match_predictions.jsonl unchanged)")
        return

    existing = {r["key"]: r for r in history.load_jsonl(PRED_PATH)}
    replaced = 0
    for rec in records:
        cur = existing.get(rec["key"])
        # force-replace only when absent or still model-null: never clobber a
        # genuinely locked pre-kickoff prediction, and stay idempotent.
        if cur is None or cur.get("model") is None:
            existing[rec["key"]] = rec
            replaced += 1
    history.write_jsonl(PRED_PATH, [existing[k] for k in sorted(existing)])
    print(f"wrote {replaced} backfilled record(s) to {PRED_PATH.name} "
          f"({len(records) - replaced} already had a locked prediction)")


if __name__ == "__main__":
    main()
