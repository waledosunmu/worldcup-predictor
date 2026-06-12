"""Kickoff-aware gate in snapshot_odds.py — controls real API spend, so guard it."""
import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest

# snapshot_odds.py lives in scripts/, not the wcpred package — load it directly.
_SPEC = importlib.util.spec_from_file_location(
    "snapshot_odds", Path(__file__).resolve().parents[1] / "scripts/snapshot_odds.py")
so = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(so)

NOW = dt.datetime(2026, 6, 20, 12, 0, tzinfo=dt.timezone.utc)


def _setup(tmp_path, commence_times, snapshot_stamps=()):
    (tmp_path / "output").mkdir()
    (tmp_path / "data/odds").mkdir(parents=True)
    (tmp_path / "output/consensus.json").write_text(json.dumps(
        {"matches": [{"home": "A", "away": "B", "commence": c} for c in commence_times]}))
    for s in snapshot_stamps:
        (tmp_path / f"data/odds/soccer_fifa_world_cup_winner_{s}.json").write_text("{}")
    so.ROOT = tmp_path


def test_snapshots_when_match_within_window(tmp_path, monkeypatch):
    kick = (NOW + dt.timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _setup(tmp_path, [kick], snapshot_stamps=["2026-06-20T000000Z"])
    monkeypatch.setattr(so, "ROOT", tmp_path)
    assert so.should_snapshot(3, NOW)[0] is True


def test_skips_in_quiet_window_with_recent_snapshot(tmp_path, monkeypatch):
    kick = (NOW + dt.timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%SZ")  # >3h away
    recent = (NOW - dt.timedelta(hours=2)).strftime("%Y-%m-%dT%H%M%SZ")   # <24h baseline
    _setup(tmp_path, [kick], snapshot_stamps=[recent])
    monkeypatch.setattr(so, "ROOT", tmp_path)
    go, reason = so.should_snapshot(3, NOW)
    assert go is False and "no match within" in reason


def test_daily_baseline_forces_snapshot_when_stale(tmp_path, monkeypatch):
    kick = (NOW + dt.timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    stale = (NOW - dt.timedelta(hours=30)).strftime("%Y-%m-%dT%H%M%SZ")  # >24h
    _setup(tmp_path, [kick], snapshot_stamps=[stale])
    monkeypatch.setattr(so, "ROOT", tmp_path)
    assert so.should_snapshot(3, NOW)[0] is True


def test_bootstrap_when_no_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(so, "ROOT", tmp_path)  # no consensus.json at all
    assert so.match_within(3, NOW) is None
    assert so.should_snapshot(3, NOW)[0] is True
