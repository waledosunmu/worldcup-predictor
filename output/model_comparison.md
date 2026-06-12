# Model comparison: v0 Poisson vs Dixon-Coles vs covariate (Groll-style)

Point-in-time: each model fit only on competitive (non-friendly) internationals before each edition's opening day, 28-year window.
Out-of-sample = the 64 matches of each World Cup 2006-2022 (320 total), scored as 3-way W/D/L. Lower is better for all metrics.

## Out-of-sample, pooled over 320 World Cup matches

| Model | Log loss | Brier | RPS | In-sample sum log-lik |
|---|---:|---:|---:|---:|
| v0_poisson | 0.9721 | 0.5671 | 0.1956 | -100345.0 |
| dixon_coles | 0.9708 | 0.5668 | 0.1955 | -100298.0 |
| covar_form | 0.9640 | 0.5611 | 0.1930 | -97688.2 |
| covar_form_host | 0.9648 | 0.5623 | 0.1937 | -97161.7 |
| hybrid_form_dc | 0.9627 | 0.5607 | 0.1929 | -97648.6 |
| dc_decay_hl2 | 0.9701 | 0.5671 | 0.1956 | n/a (weighted) |
| dc_decay_hl4 | 0.9700 | 0.5668 | 0.1955 | n/a (weighted) |
| dc_decay_hl8 | 0.9700 | 0.5667 | 0.1954 | n/a (weighted) |
| dc_decay_hl16 | 0.9703 | 0.5667 | 0.1954 | n/a (weighted) |

Best time-decay half-life by OOS log loss: dc_decay_hl4.

The in-sample column sums each edition's pre-opening fit-window log-likelihood. It is comparable across the non-decay models (same data, more parameters -> higher is better) but NOT across decay half-lives (different weights change the LL scale).

## Per edition, RPS

| Edition | v0_poisson | dixon_coles | covar_form | covar_form_host | hybrid_form_dc | dc_decay_hl2 | dc_decay_hl4 | dc_decay_hl8 | dc_decay_hl16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2006 | 0.1687 | 0.1685 | 0.1596 | 0.1599 | 0.1594 | 0.1689 | 0.1685 | 0.1684 | 0.1684 |
| 2010 | 0.1902 | 0.1899 | 0.1856 | 0.1852 | 0.1853 | 0.1904 | 0.1902 | 0.1901 | 0.1900 |
| 2014 | 0.1837 | 0.1838 | 0.1811 | 0.1843 | 0.1812 | 0.1849 | 0.1847 | 0.1843 | 0.1841 |
| 2018 | 0.2065 | 0.2066 | 0.2043 | 0.2032 | 0.2044 | 0.2067 | 0.2066 | 0.2066 | 0.2066 |
| 2022 | 0.2286 | 0.2285 | 0.2342 | 0.2358 | 0.2342 | 0.2273 | 0.2275 | 0.2278 | 0.2281 |

## Covariate availability (Groll et al. full hybrid)

| Covariate | Available here | Notes / data needed |
|---|:--:|---|
| form_diff | yes | rolling recent goal-difference-per-match, home minus away; source: data/raw/results.csv (computed point-in-time in build_form_frame) |
| host | yes | home team in its own country; redundant with dr home-adv; source: data/raw/results.csv neutral/country columns |
| market_value_diff | no | log total squad market value, home minus away (Groll's top predictor); source: transfermarkt squad valuations per tournament -- NOT in repo |
| squad_age_diff | no | mean squad age, home minus away; source: per-tournament squad lists with birthdates -- NOT in repo |
| cl_players_diff | no | # players at Champions-League clubs, home minus away; source: squad-to-club mapping + UCL participant list -- NOT in repo |
| confederation | no | team confederation (non-differential; needs interaction terms); source: team->confederation map; derivable but leakage-prone, deferred |

Covariates marked 'no' require external data not shipped in this repo and are scaffolded (COVARIATE_SPEC + the symmetric design-matrix interface in covariates.fit) but never fabricated.
