"""Live-results overlay: format-preserving write + NA-row fill.

Regression guard for the bug where writing back results.csv reformatted all
~49k rows (float `.0` scores, `True`/`False` casing, empty cells for `NA`),
producing a multi-MB diff every daily run instead of touching only the rows
whose scores changed.
"""
import pandas as pd

from wcpred import live

# Mimics the martj42 snapshot: integer scores, literal "NA" for unplayed,
# uppercase TRUE/FALSE for neutral.
SNAPSHOT = (
    "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
    "1872-11-30,Scotland,England,0,0,Friendly,Glasgow,Scotland,FALSE\n"
    "2026-06-11,Mexico,South Africa,NA,NA,FIFA World Cup,Mexico City,Mexico,FALSE\n"
    "2026-06-11,South Korea,Czech Republic,NA,NA,FIFA World Cup,Zapopan,Mexico,TRUE\n"
)


def test_write_results_csv_preserves_format_when_unchanged(tmp_path):
    src = tmp_path / "results.csv"
    src.write_text(SNAPSHOT)
    df = pd.read_csv(src)  # NA -> float scores, neutral -> bool
    out = tmp_path / "out.csv"
    live.write_results_csv(df, out)
    # Round-trip must be byte-identical to the snapshot: no .0, no False, no empty NA.
    assert out.read_text() == SNAPSHOT


def test_write_results_csv_renders_filled_score_as_int(tmp_path):
    df = pd.read_csv(pd.io.common.StringIO(SNAPSHOT))
    df.loc[df.home_team == "Mexico", ["home_score", "away_score"]] = [2, 0]
    out = tmp_path / "out.csv"
    live.write_results_csv(df, out)
    lines = out.read_text().splitlines()
    mex = next(l for l in lines if l.startswith("2026-06-11,Mexico"))
    assert mex == "2026-06-11,Mexico,South Africa,2,0,FIFA World Cup,Mexico City,Mexico,FALSE"
    # untouched rows still carry literal NA, not an empty cell
    assert any(l.endswith("Zapopan,Mexico,TRUE") and ",NA,NA," in l for l in lines)


def test_merge_results_fills_na_orientation_robust():
    results = pd.read_csv(pd.io.common.StringIO(SNAPSHOT))
    shootouts = pd.DataFrame(columns=["date", "home_team", "away_team", "winner", "first_shooter"])
    valid = {"Mexico", "South Africa", "South Korea", "Czech Republic"}
    # feed delivers one match with teams in the OPPOSITE orientation to the snapshot
    matches = [
        {"home": "South Africa", "away": "Mexico", "home_score": 0, "away_score": 2,
         "pen_home": None, "pen_away": None, "date": "2026-06-11"},
    ]
    merged, _, summary = live.merge_results(results, shootouts, matches, valid)
    assert summary["filled"] == 1 and not summary["unmapped"]
    row = merged[merged.home_team == "Mexico"].iloc[0]
    # score re-oriented onto the snapshot's home/away (Mexico 2-0), not flipped
    assert row.home_score == 2 and row.away_score == 0
