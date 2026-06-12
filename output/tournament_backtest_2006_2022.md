# Tournament-level backtest: World Cups 2006-2022 (point-in-time, whole-tournament MC)

Each edition simulated as a full 32-team tournament (50,000 Monte Carlo runs) with the model refit on ONLY pre-tournament data — same Elo + 28-year Poisson fit window as the match-level backtest. Real format/draw reconstructed per edition (8 groups of 4 from fixture cliques; fixed R16 bracket and KO tree from actual matchups — a pre-tournament structural fact, no result leakage).

## Champion skill (lower log loss / Brier is better)

| Metric | elo_poisson | elo_expectancy (baseline) |
|---|---:|---:|
| Champion log loss (mean over editions) | 0.0930 | 0.0932 |
| Champion Brier (mean over editions) | 0.02568 | 0.02577 |
| Mean P assigned to the actual champion | 0.162 | — |

Pooled over all 160 (team, edition) champion outcomes: log loss 0.0930, Brier 0.02568 (elo_poisson). Champion skill is only 5 outcomes — low statistical power; read it alongside the richer reach-final calibration below.

## Per edition (elo_poisson)

| Edition | Host | Champion | P(champion) | P(finalists, avg) | Champ Brier | Top-3 by P(champion) |
|---|---|---|---:|---:|---:|---|
| 2006 | Germany | Italy | 4.8% | 14.0% | 0.0308 | Brazil 16%, England 10%, Netherlands 10% |
| 2010 | South Africa | Spain | 30.8% | 35.7% | 0.0175 | Spain 31%, Brazil 22%, Netherlands 13% |
| 2014 | Brazil | Germany | 16.7% | 24.9% | 0.0258 | Spain 25%, Brazil 24%, Germany 17% |
| 2018 | Russia | France | 5.5% | 7.5% | 0.0326 | Brazil 31%, Germany 14%, Spain 14% |
| 2022 | Qatar | Argentina | 23.3% | 23.3% | 0.0216 | Brazil 29%, Argentina 23%, Netherlands 8% |

## Reach-the-final calibration (elo_poisson, 160 team-edition outcomes, 10 finalists)

Pooled reach-final log loss 0.1658, Brier 0.04873; mean P assigned to the 10 actual finalists 0.211.

| Predicted P(final) bin | n | mean predicted | observed freq |
|---|---:|---:|---:|
| [0.00,0.05) | 107 | 0.014 | 0.009 |
| [0.05,0.10) | 22 | 0.077 | 0.045 |
| [0.10,0.20) | 18 | 0.143 | 0.167 |
| [0.20,0.40) | 11 | 0.302 | 0.364 |
| [0.40,1.01) | 2 | 0.453 | 0.500 |

## Descriptive baseline: pre-tournament Elo #1 favorite

Across 5 editions, the single highest pre-tournament Elo team reached the final 1 time(s) and won 1 time(s).

## Known approximations

- Group draws reconstructed as 4-team cliques over the 48 group fixtures; group letters A-H assigned by sorted clique order (letters are cosmetic — the bracket template binds slots to real groups, so letter choice does not affect results).
- R16 bracket and KO tree recovered from the actual qualifier matchups and which winners met next. This is the pre-tournament draw structure (fixed before kickoff); only the bracket TOPOLOGY is taken from reality, never any match outcome the model is asked to predict.
- LEGACY (pre-2026) group tiebreakers: points -> overall GD -> overall goals -> head-to-head; residual ties broken by pre-tournament Elo (proxy for drawing of lots). Conduct/fair-play tiebreaker not simulated.
- Host nation (Germany'06, South Africa'10, Brazil'14, Russia'18, Qatar'22) gets the +100 Elo home advantage in the group stage only; knockouts are neutral for everyone (matches the 2026 simulator's convention).
- elo_expectancy baseline shares the identical bracket AND Poisson group scoring; the two models differ only in the knockout win function (Poisson scoreline vs raw Elo expected result). The champion log loss barely separates them (0.0930 vs 0.0932) because the group stage is shared and the binary champion event has only 5 outcomes — the goals model adds little over raw Elo on this particular metric at n=5.
- Knockout 'draw' resolution uses the dataset convention (ET included in the score, shootouts from shootouts.csv) when recovering real winners; simulated knockouts use ET at 0.33x expected goals then a coin flip.
