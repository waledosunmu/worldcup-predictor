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
- [x] Backtest harness 2006–2022, point-in-time refit per edition
      ([output/backtest_2006_2022.md](output/backtest_2006_2022.md)):
      elo_poisson beats both baselines on log loss (0.972 vs 1.021/1.067)
      and ties the Elo-expectancy baseline on RPS (0.196)
- [x] Bookmaker-consensus model (Leitner overround removal, geometric-mean
      pooling): 5 full-field outright books, 72 match markets (up to 49 books
      per match) snapshotted pre-kickoff on opening day
- [x] Explanation generator: templated, fully traceable narratives per fixture
      (Elo gap, xG, market divergence, head-to-head, recent form)
- [x] Static site → [docs/](docs/) (GitHub Pages-ready): forecast, all 72
      fixtures with reasoning, methodology + backtest
- [x] Conditional re-forecasting (`scripts/run_forecast.py --as-of`): played
      group scores and knockout winners (incl. shootouts) locked into every
      simulation, Elo updated through the latest result; verified vs synthetic
      results and reproduces the opening forecast when nothing has been played
- [ ] Hybrid model with covariates (Groll et al. style) + Dixon-Coles
- [ ] Tournament-level backtest (champion distributions for 2006–2022)

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install pandas numpy scipy
cp .env.example .env                                  # add your ODDS_API_KEY
.venv/bin/python scripts/run_forecast.py              # conditional on played results
.venv/bin/python scripts/run_backtest.py              # 2006-2022 validation
.venv/bin/python scripts/snapshot_odds.py             # capture odds (daily!)
.venv/bin/python scripts/run_consensus.py             # bookmaker consensus
.venv/bin/python scripts/build_site.py                # render docs/
bash scripts/update_all.sh                            # ...or all of the above
open docs/index.html
```

Hosting: push to GitHub and enable Pages (Settings → Pages → deploy from
branch, `/docs` folder). Netlify/Cloudflare Pages also work — point them at
`docs/`.

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
