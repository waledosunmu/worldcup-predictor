# Roadmap

Living list of what's shipped and what's next. Dates are absolute.

## Shipped
- **Opening-day v0** — Elo engine (r=0.986 vs eloratings.net), Poisson goals model,
  100k Monte Carlo sim with 2026 rules.
- **Backtests** — match-level 2006–2022 (beats baselines on log loss) and
  whole-tournament champion-distribution backtest.
- **Bookmaker consensus** — overround removal + geometric pooling; model-vs-market
  comparison.
- **Live results** — football-data.org overlay onto the martj42 snapshot with a
  format-preserving writer; martj42 is the fallback.
- **Hybrid model (research)** — Dixon-Coles low-score correction + recent-form
  covariate; backtested point-in-time (`output/model_comparison.md`). v0 stays the
  production default until the gains justify promotion.
- **Probability history** — `probability_history.jsonl` (trend graph + movers) and
  `match_predictions.jsonl` (live model-vs-market track record).
- **Kickoff-aware odds cadence** — `snapshot_odds.py` only spends API credits near a
  kickoff (or a daily baseline); the 3-hourly workflow keeps the site fresh cheaply.
- **Odds-key fix** — odds requests are restricted to the men's 2026 World Cup markets
  (no qualifiers / Club / Women's / cricket World Cups).
- **Odds-movement tracker** — `output/odds_movement.md`: how much the market drifts
  before kickoff, to decide whether the cadence is worth its cost.
- **Interactive site** — sortable team table, championship trend graph, model-vs-market
  divergence, interactive knockout bracket, per-team detail modal, movers + live
  track-record (`docs/`, Chart.js vendored, no build step).

## Near-term
- **Tune the odds cadence from data.** After the first few matchdays of the 3-hourly
  schedule, read `output/odds_movement.md`. If late (final-3h) movement is negligible,
  scale down: widen `ODDS_SNAPSHOT_WINDOW_H`, raise `ODDS_SNAPSHOT_BASELINE_H`, or
  lengthen the workflow cron. The free Odds API tier is ~500 credits/month and each
  snapshot costs ~6, so this directly governs whether we stay within quota.
- **Promote the hybrid model** if it keeps beating v0 out-of-sample as results land.

## Later
- **Groll-style squad covariates** (market value, squad age, Champions-League minutes) —
  the strongest missing predictors; need external per-tournament squad data, scaffolded
  behind `COVARIATE_SPEC`.
- **Scenario / what-if** — let users fix a hypothetical result and see updated
  probabilities (client-side conditional or precomputed).
- **Finer intraday odds series** — if the movement tracker shows meaningful late drift,
  keep a within-day series (not just per-`as_of`) for closing-line-value analysis.

## Odds storage policy
Raw match-odds snapshots (~1.4 MB each) are gitignored
(`data/odds/soccer_fifa_world_cup_2026*.json`) — they are an ephemeral API cache. The
committed sources of truth are the compact derived artifacts: `output/consensus.json`,
`output/odds_timeseries.jsonl`, the tiny `*_winner_*` outright snapshots, and the
history files. The pipeline derives everything it keeps in the same run that fetches the
raw odds.
