"""World Football Elo engine, replicating the published eloratings.net methodology.

K by match importance (eloratings.net/about): 60 World Cup finals; 50 continental
finals & major intercontinental; 40 qualifiers & major tournaments; 30 other
tournaments; 20 friendlies. Margin-of-victory multiplier: x1.5 for 2-goal wins,
x1.75 for 3, x(1.75 + (N-3)/8) for N>=4. Expectancy We = 1/(10**(-dr/400) + 1)
with dr including +100 home advantage for non-neutral venues.

Note: We is the expected RESULT (draw counts 0.5), not a pure win probability.
Ratings here are computed from scratch (all teams start at 1500) over the full
martj42 results history, so absolute values differ from eloratings.net's
(which seeded historical baselines), but differences dr — the only quantity the
models consume — track closely.
"""

import pandas as pd

BASE_RATING = 1500.0
HOME_ADV = 100.0

CONTINENTAL_FINALS = {
    "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
    "CONCACAF Championship", "Gold Cup", "Oceania Nations Cup",
    "FIFA Confederations Cup",
}
MAJOR_40 = {"UEFA Nations League", "CONCACAF Nations League"}


def k_factor(tournament: str) -> float:
    if tournament == "FIFA World Cup":
        return 60.0
    if tournament in CONTINENTAL_FINALS:
        return 50.0
    if "qualification" in tournament or tournament in MAJOR_40:
        return 40.0
    if tournament == "Friendly":
        return 20.0
    return 30.0


def mov_multiplier(goal_diff: int) -> float:
    n = abs(goal_diff)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    if n == 3:
        return 1.75
    return 1.75 + (n - 3) / 8.0


def expectancy(dr: float) -> float:
    """Expected result for the side whose rating advantage is dr (incl. home adv)."""
    return 1.0 / (10.0 ** (-dr / 400.0) + 1.0)


def compute_ratings(results: pd.DataFrame, as_of: str | None = None) -> dict[str, float]:
    """Run the Elo engine over all completed matches before `as_of` (exclusive).

    `results` is the martj42 results.csv frame. Returns {team: rating}.
    """
    df = results.dropna(subset=["home_score", "away_score"])
    if as_of is not None:
        df = df[df["date"] < as_of]
    ratings: dict[str, float] = {}
    cols = ["home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]
    for home, away, hs, as_, tourn, neutral in df[cols].itertuples(index=False):
        rh = ratings.get(home, BASE_RATING)
        ra = ratings.get(away, BASE_RATING)
        dr = rh - ra + (0.0 if neutral else HOME_ADV)
        we = expectancy(dr)
        w = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        k = k_factor(tourn) * mov_multiplier(int(hs - as_))
        delta = k * (w - we)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    return ratings


def rating_history_frame(results: pd.DataFrame) -> pd.DataFrame:
    """Per-match pre-match ratings/diff for every completed match — input for
    fitting the goals model with point-in-time discipline."""
    df = results.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    rows = []
    cols = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]
    for date, home, away, hs, as_, tourn, neutral in df[cols].itertuples(index=False):
        rh = ratings.get(home, BASE_RATING)
        ra = ratings.get(away, BASE_RATING)
        dr = rh - ra + (0.0 if neutral else HOME_ADV)
        rows.append((date, home, away, int(hs), int(as_), tourn, bool(neutral), rh, ra, dr))
        we = expectancy(dr)
        w = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        k = k_factor(tourn) * mov_multiplier(int(hs - as_))
        delta = k * (w - we)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    return pd.DataFrame(rows, columns=[
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "neutral", "elo_home_pre", "elo_away_pre", "dr",
    ])
