# CLAUDE.md — worldcup-predictor

2026 FIFA World Cup forecaster: Elo → Poisson goals → Monte Carlo, validated by
point-in-time backtesting against 2006–2022, published as a static site.

This file is the shared contract for the agent framework in `.claude/agents/`.
Every agent reads it. If a convention here conflicts with an agent prompt, this
file wins.

## Layout

| Path | What lives here | Edit rules |
|------|-----------------|-----------|
| `src/wcpred/` | The library: pure-ish modeling modules (`elo`, `goals`, `dixoncoles`, `covariates`, `simulate`, `odds`, `consensus` math, `calibration`, `metrics`, `explain`, `history`, `live`, `odds_movement`). | All new modeling logic goes here. Keep modules small (current max ~270 LOC). |
| `scripts/` | Runnable pipeline entrypoints (`run_forecast`, `run_backtest`, `run_tournament_backtest`, `run_model_comparison`, `snapshot_odds`, `fetch_results`, `run_consensus`, `append_history`, `build_site`, `update_all.sh`). | Thin orchestration over `src/`. No business logic that can't be unit-tested in `src/`. |
| `tests/` | Flat pytest files, `test_<module>.py`. | One test file per module/concern. No `tests/unit/` subdir. |
| `output/` | Committed derived artifacts (`*.json` + `*.md`): forecasts, backtests, model comparison, consensus, history (`*.jsonl`). | The committed source of truth for results. Regenerate, don't hand-edit. |
| `docs/` | Static site (GitHub Pages), Chart.js vendored, no build step. Rebuilt by `build_site.py` from `output/`. | Don't hand-edit generated HTML/`data.json`; change `build_site.py`. |
| `data/raw` | Snapshotted inputs (martj42 `results.csv`, eloratings TSVs, `format_2026.json`). | Treat as immutable point-in-time snapshots. |
| `data/odds` | Ephemeral raw odds cache (gitignored). | Never commit; never required for tests. |

## Commands

```bash
uv run pytest                          # full suite (pyproject sets pythonpath=src)
uv run pytest tests/test_<module>.py   # one file
.venv/bin/python scripts/run_forecast.py
.venv/bin/python scripts/run_backtest.py
.venv/bin/python scripts/run_model_comparison.py
```

- Runtime deps: `pandas numpy scipy scikit-learn`. Test dep: `pytest`. **That is all.**
- **No ruff / pyright / mypy / pandera are installed.** Do not write tests or gates
  that import them. If you want a lint/type pass, it is optional and ad hoc via
  `uvx ruff check src/` / `uvx pyright src/` — never a required gate, never assumed present.
- Python ≥ 3.11. Prefer stdlib typing; full annotations on public functions are
  encouraged but not enforced by a checker.

## Cardinal rules (get these wrong and every metric silently lies)

1. **Point-in-time / leak-free.** A forecast or backtest for date *T* may use only
   information available before *T*. Backtests refit the model per edition using
   only prior matches; brackets are reconstructed from fixtures, never from known
   outcomes. Never let a future result, final Elo, or post-tournament rating leak
   into a pre-*T* computation. This is the analog of a production-data write: it
   won't fail a unit test, so it must be asserted explicitly and reviewed deliberately.
2. **Determinism.** Monte Carlo (`simulate.py`, 100k sims) must be seedable and
   reproducible. Thread an explicit RNG/seed; never rely on global `np.random`
   state. A forecast run with nothing newly played must reproduce the prior forecast.
3. **No fabricated data.** Missing inputs (squad market value, CL minutes, squad age)
   stay scaffolded behind `COVARIATE_SPEC` — never invent numbers to fill them.
   If a source is absent, the code path is disabled, not faked.
4. **No live API calls in dev or tests.** Odds (The Odds API, ~500 credits/month,
   ~6/snapshot) and football-data.org are budgeted and rate-limited. Tests and
   local iteration use committed snapshots / fixtures / `tmp_path` + monkeypatch
   (see `tests/test_snapshot_gate.py`). Only an explicit, human-run script spends credits.

## Validation model

This is a forecasting repo, so "correct" has two meanings:

- **Deterministic logic** (Elo replication, 2026 format/tiebreakers, leak-free
  bracket reconstruction, results overlay, odds/consensus math, the spend gate) is
  validated by **unit tests** — red/green TDD applies.
- **Modeling changes** ("does the hybrid beat v0 out-of-sample?") have no failing
  unit test to write. They are validated by **backtest + calibration**: match-level
  log loss / RPS (`output/backtest_2006_2022.md`, `output/model_comparison.md`) and
  tournament-level champion/finalist calibration (`output/tournament_backtest_2006_2022.md`).
  A modeling change ships only if it does not regress these on committed baselines.

## Parallel-agent hygiene

Agents may run in separate worktrees. `tests/test_imports_worktree_src.py` guards
against a sibling checkout shadowing `src/`. Keep tasks scoped to one module where
possible so parallel implementers don't collide.
