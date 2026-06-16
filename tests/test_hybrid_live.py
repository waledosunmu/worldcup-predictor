"""Promoting the recent-form + Dixon-Coles hybrid into the live simulator.

Deterministic plumbing only (per CLAUDE.md): that current form is leak-free, and
that the simulator's expected-goals path reduces to v0 without covariates and uses
the covariate-augmented means with them. Whether the hybrid *predicts better* is a
backtest question, not a unit test.
"""
import json
from pathlib import Path

import pandas as pd

from wcpred import covariates, goals
from wcpred.simulate import Simulator

ROOT = Path(__file__).resolve().parents[1]

FMT = {  # minimal 1-group, 3-team format (enough to build group fixtures)
    "groups": {"A": ["Alpha", "Beta", "Gamma"]},
    "round_of_32": {}, "round_of_16": {}, "quarterfinals": {},
    "semifinals": {}, "final": {},
}


def _hist(rows):
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                       "home_score", "away_score"])


def test_current_form_is_leak_free_and_correct():
    hist = _hist([
        ["2024-01-01", "Alpha", "Beta", 3, 0],   # Alpha +3, Beta -3
        ["2024-02-01", "Alpha", "Gamma", 1, 1],  # Alpha 0, Gamma 0
        ["2026-07-01", "Alpha", "Beta", 9, 0],   # AFTER as_of — must be ignored
    ])
    form = covariates.current_form(hist, window=5, as_of="2026-06-15")
    assert form["Alpha"] == (3 + 0) / 2          # mean of +3 and 0; future game excluded
    assert form["Beta"] == -3.0
    assert "Gamma" in form and form["Gamma"] == 0.0


def test_simulator_means_reduce_to_v0_without_covariates():
    ratings = {"Alpha": 1900.0, "Beta": 1800.0, "Gamma": 1700.0}
    params = {"a": 0.2, "b": 0.78}                # no 'beta' -> v0 path
    sim = Simulator(FMT, ratings, params, seed=1)
    # every group fixture's stored means equal goals.expected_goals(dr)
    for g, home, away, lh, la in sim.group_fixtures:
        dr = ratings[home] - ratings[away]         # no hosts in this fixture set
        elh, ela = goals.expected_goals(dr, params)
        assert (lh, la) == (elh, ela)


def test_simulator_means_use_covariates_when_hybrid():
    ratings = {"Alpha": 1900.0, "Beta": 1800.0, "Gamma": 1700.0}
    params = {"a": 0.2, "b": 0.64, "rho": -0.03,
              "beta": {"form_diff": 0.08}, "covariates": ["form_diff"]}
    team_form = {"Alpha": 2.0, "Beta": -1.0, "Gamma": 0.0}
    sim = Simulator(FMT, ratings, params, seed=1, dixon_coles=True, team_form=team_form)
    for g, home, away, lh, la in sim.group_fixtures:
        dr = ratings[home] - ratings[away]
        z = {"form_diff": team_form[home] - team_form[away]}
        elh, ela = covariates.expected_goals(dr, z, params)
        assert (lh, la) == (elh, ela)
    # covariate means must differ from the Elo-only means (form_diff != 0 here)
    v0 = Simulator(FMT, ratings, {"a": 0.2, "b": 0.64}, seed=1)
    assert sim.group_fixtures[0][3] != v0.group_fixtures[0][3]


def test_hybrid_simulation_is_deterministic():
    # full 2026 format so the real bracket runs end to end
    fmt = json.load(open(ROOT / "data/format_2026.json"))
    teams = [t for g in fmt["groups"].values() for t in g]
    ratings = {t: 1700.0 + 4.0 * i for i, t in enumerate(teams)}
    params = {"a": 0.2, "b": 0.64, "rho": -0.03, "beta": {"form_diff": 0.08},
              "covariates": ["form_diff"]}
    tf = {t: (i % 5) - 2.0 for i, t in enumerate(teams)}
    a = Simulator(fmt, ratings, params, seed=7, dixon_coles=True, team_form=tf).run(500)
    b = Simulator(fmt, ratings, params, seed=7, dixon_coles=True, team_form=tf).run(500)
    assert a["probs"] == b["probs"]              # same seed -> identical
