# worldcup-predictor

Predicting every match of the 2026 FIFA World Cup — group stages, knockout
rounds, and the champion — with models validated by backtesting against past
World Cups, published with per-prediction reasoning.

## Status (opening day, 2026-06-11)

- [x] Data layer: martj42 results (49,477 matches, snapshot committed for
      point-in-time discipline), eloratings.net TSVs, football-data.org planned
      for live results
- [x] 2026 format encoded ([data/format_2026.json](data/format_2026.json)):
      12 groups verified two ways (Wikipedia per-group pages + fixture cliques),
      R32 bracket M73–M104, new tiebreaker order
- [x] Elo engine replicating eloratings.net methodology (corr 0.986 with
      official ratings across the 48 squads)
- [x] Poisson goals model (Elo diff → expected goals, MLE on ~17.9k competitive
      internationals since 1998)
- [x] Monte Carlo simulator with 2026 rules (100k sims ≈ 10 s)
- [x] Opening-day forecast: [output/forecast_opening.md](output/forecast_opening.md)
- [x] Odds snapshot (partial manual capture + The Odds API script)
- [ ] Backtest harness 2006–2022 (Brier / log loss / RPS vs Elo-only baseline)
- [ ] Bookmaker-consensus model (Leitner/Zeileis/Hornik overround recipe)
- [ ] Hybrid model with covariates (Groll et al. style)
- [ ] Explanation generator (per-match factors → narrative)
- [ ] Static site (model JSON → daily rebuild)

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install pandas numpy scipy
.venv/bin/python scripts/run_opening_forecast.py --sims 100000
ODDS_API_KEY=... .venv/bin/python scripts/snapshot_odds.py   # daily!
```

## Architecture

```
data/raw (results.csv, elo TSVs)  →  src/wcpred/elo.py    (point-in-time ratings)
                                  →  src/wcpred/goals.py  (Elo diff → Poisson xG)
data/format_2026.json             →  src/wcpred/simulate.py (Monte Carlo, 2026 rules)
                                  →  output/forecast_*.{json,md}
```

Methodology follows the published academic recipe (Groll et al. JQAS 2019;
Leitner/Zeileis/Hornik IJF 2010): Poisson scores from expected goals, official
group rules, extra time at 0.33× expected goals, penalties as coin flip,
100,000 tournament replications.

## Known approximations (v0)

- Third-place R32 slots: constraint-satisfying matching over slot eligibility
  sets, not FIFA's exact 495-combination table.
- Conduct-score (cards) tiebreaker not simulated; FIFA-ranking tiebreaker
  proxied by pre-tournament Elo.
- Hosts get +100 Elo only in the group stage; knockouts treated as neutral.
- Elo engine starts all teams at 1500 in 1872 rather than eloratings.net's
  seeded baselines; differences (the model input) track officially published
  ratings at r = 0.986.
- No squad-level covariates yet (market value, club form) — planned for the
  hybrid model.
