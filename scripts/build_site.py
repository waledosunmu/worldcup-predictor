"""Render the prediction site to docs/ (GitHub Pages-ready) from model artifacts.

Inputs: output/forecast_opening.json, output/consensus.json,
        output/backtest_2006_2022.json, data/format_2026.json, data/raw/results.csv
Usage:  python scripts/build_site.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpred.explain import fixture_explanation, team_blurb

AS_OF = "2026-06-11"
DOCS = ROOT / "docs"

CSS = """
:root { --bg:#fafafa; --card:#fff; --ink:#1a1a2e; --muted:#6b7280;
        --model:#2563eb; --market:#d97706; --line:#e5e7eb; }
* { box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:var(--bg); color:var(--ink); margin:0; line-height:1.55; }
main { max-width:920px; margin:0 auto; padding:1.5rem 1rem 4rem; }
header { background:var(--ink); color:#fff; padding:1.6rem 1rem; }
header .inner { max-width:920px; margin:0 auto; }
header h1 { margin:0 0 .2rem; font-size:1.6rem; }
header p, header a { color:#cbd5e1; margin:0; }
nav a { color:#fff; margin-right:1.2rem; text-decoration:none; font-weight:600; }
nav { margin-top:.8rem; }
h2 { margin-top:2.2rem; border-bottom:2px solid var(--line); padding-bottom:.3rem; }
table { border-collapse:collapse; width:100%; background:var(--card);
        font-size:.92rem; }
th, td { text-align:left; padding:.45rem .6rem; border-bottom:1px solid var(--line); }
th { background:#f1f5f9; } td.num, th.num { text-align:right; }
.bar { background:var(--line); border-radius:3px; height:10px; min-width:120px; }
.bar > div { height:10px; border-radius:3px; }
.bar .model { background:var(--model); } .bar .market { background:var(--market); }
.legend span { display:inline-block; margin-right:1rem; font-size:.85rem;
               color:var(--muted); }
.dot { display:inline-block; width:10px; height:10px; border-radius:5px;
       margin-right:.3rem; }
.fixture { background:var(--card); border:1px solid var(--line); border-radius:8px;
           padding: .8rem 1rem; margin:.7rem 0; }
.fixture .teams { font-weight:650; }
.fixture .date { color:var(--muted); font-size:.85rem; float:right; }
.probrow { display:flex; gap:.4rem; margin:.4rem 0; font-size:.88rem; }
.probrow span { flex:1; text-align:center; padding:.25rem; border-radius:4px;
                background:#eef2ff; }
.probrow span.fav { background:#dbeafe; font-weight:650; }
details summary { cursor:pointer; color:var(--model); font-size:.88rem; }
details p { font-size:.9rem; color:#374151; }
.muted { color:var(--muted); font-size:.85rem; }
footer { color:var(--muted); font-size:.8rem; margin-top:3rem;
         border-top:1px solid var(--line); padding-top:1rem; }
"""


def page(title: str, body: str, updated: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — WC26 Predictor</title><link rel="stylesheet" href="style.css"></head>
<body><header><div class="inner"><h1>⚽ World Cup 2026 Predictor</h1>
<p>Probabilistic forecasts with the reasoning behind every prediction.</p>
<nav><a href="index.html">Forecast</a><a href="groups.html">Groups &amp; matches</a>
<a href="methodology.html">Methodology</a></nav></div></header>
<main>{body}<footer>Updated {updated}. Model: Elo–Poisson v0, 100k Monte Carlo
simulations. Market: bookmaker consensus via The Odds API. Data: martj42
international results, eloratings.net. Probabilities are estimates, not advice.
</footer></main></body></html>"""


def bar(p: float, cls: str, scale: float = 1.0) -> str:
    return (f'<div class="bar"><div class="{cls}" '
            f'style="width:{min(100, p / scale * 100):.1f}%"></div></div>')


def probrow(ph, pd_, pa, h, a) -> str:
    fav = 0 if ph >= max(pd_, pa) else (1 if pd_ >= pa else 2)
    cells = [f"{h} {ph:.0%}", f"Draw {pd_:.0%}", f"{a} {pa:.0%}"]
    return ('<div class="probrow">'
            + "".join(f'<span class="{"fav" if i == fav else ""}">{c}</span>'
                      for i, c in enumerate(cells)) + "</div>")


def main():
    opening = json.load(open(ROOT / "output/forecast_opening.json"))
    latest_path = ROOT / "output/forecast_latest.json"
    forecast = json.load(open(latest_path)) if latest_path.exists() else opening
    as_of = forecast.get("as_of", AS_OF)
    consensus = json.load(open(ROOT / "output/consensus.json"))
    backtest = json.load(open(ROOT / "output/backtest_2006_2022.json"))
    fmt = json.load(open(ROOT / "data/format_2026.json"))
    results = pd.read_csv(ROOT / "data/raw/results.csv")

    ratings = forecast["ratings"]
    adv = forecast["advancement"]
    ranks = {t: i + 1 for i, (t, _) in
             enumerate(sorted(ratings.items(), key=lambda kv: -kv[1]))}
    team_group = {t: g for g, ts in fmt["groups"].items() for t in ts}
    market_champ = consensus["outright_consensus"]
    market_by_fixture = {(m["home"], m["away"]): m for m in consensus["matches"]}
    max_champ = max(max(a["champion"] for a in adv.values()),
                    max(market_champ.values()))

    # ---------- index ----------
    opening_champ = {t: opening["advancement"][t]["champion"]
                     for t in opening["advancement"]}
    rows = sorted(adv.items(), key=lambda kv: -kv[1]["champion"])[:16]
    champ_rows = ""
    for t, p in rows:
        mk = market_champ.get(t, 0.0)
        blurb = team_blurb(t, p, ratings[t], ranks[t], team_group[t], mk)
        champ_rows += (
            f"<tr><td><b>{t}</b><br><span class='muted'>{blurb}</span></td>"
            f"<td class='num'>{p['champion']:.1%}{bar(p['champion'], 'model', max_champ)}</td>"
            f"<td class='num'>{mk:.1%}{bar(mk, 'market', max_champ)}</td>"
            f"<td class='num'>{opening_champ.get(t, 0):.1%}</td>"
            f"<td class='num'>{p['reach_final']:.0%}</td>"
            f"<td class='num'>{p['reach_sf']:.0%}</td></tr>")

    upcoming = sorted(consensus["matches"], key=lambda m: m["commence"])
    up_html, shown = "", 0
    for m in upcoming:
        fx = next((f for f in forecast["group_fixtures"]
                   if f["home"] == m["home"] and f["away"] == m["away"]), None)
        if not fx or fx.get("played") or shown >= 9:
            continue
        shown += 1
        expl = fixture_explanation(fx, ratings, ranks, results, as_of, m)
        up_html += (
            f'<div class="fixture"><span class="date">{m["commence"][:10]} · '
            f'Group {fx["group"]}</span>'
            f'<span class="teams">{m["home"]} vs {m["away"]}</span>'
            + probrow(fx["p_home"], fx["p_draw"], fx["p_away"], m["home"], m["away"])
            + f'<details><summary>Why the model thinks this</summary><p>{expl}</p>'
              f'</details></div>')

    index_body = f"""
<h2>Who wins the 2026 World Cup?</h2>
<p class="legend"><span><i class="dot" style="background:var(--model)"></i>
Model (Elo–Poisson, 100k sims)</span><span><i class="dot"
style="background:var(--market)"></i>Bookmaker consensus
({consensus['n_outright_books']} books)</span></p>
<table><tr><th>Team</th><th class="num">Champion (model)</th>
<th class="num">Champion (market)</th><th class="num">Pre-tournament</th>
<th class="num">Final</th><th class="num">Semis</th></tr>{champ_rows}</table>
<h2>Next matches</h2>{up_html}
<p><a href="groups.html">All 72 group fixtures with predictions →</a></p>"""

    # ---------- groups ----------
    fixtures_by_group = {}
    for fx in forecast["group_fixtures"]:
        fixtures_by_group.setdefault(fx["group"], []).append(fx)
    groups_body = "<h2>Groups &amp; all fixtures</h2>"
    for g in sorted(fmt["groups"]):
        trows = ""
        for t in sorted(fmt["groups"][g], key=lambda t: -adv[t]["reach_r32"]):
            p = adv[t]
            trows += (f"<tr><td>{t}</td><td class='num'>{ratings[t]:.0f}</td>"
                      f"<td class='num'>{p['win_group']:.0%}</td>"
                      f"<td class='num'>{p['reach_r32']:.0%}</td>"
                      f"<td class='num'>{p['reach_qf']:.0%}</td>"
                      f"<td class='num'>{p['champion']:.1%}</td></tr>")
        fxs = ""
        for fx in fixtures_by_group[g]:
            m = market_by_fixture.get((fx["home"], fx["away"]))
            date = m["commence"][:10] if m else ""
            if fx.get("played"):
                fxs += (f'<div class="fixture"><span class="date">{date} · '
                        f'final</span><span class="teams">{fx["home"]} '
                        f'{fx["score_home"]}–{fx["score_away"]} {fx["away"]}'
                        f'</span></div>')
                continue
            expl = fixture_explanation(fx, ratings, ranks, results, as_of, m)
            fxs += (f'<div class="fixture"><span class="date">{date}</span>'
                    f'<span class="teams">{fx["home"]} vs {fx["away"]}</span>'
                    + probrow(fx["p_home"], fx["p_draw"], fx["p_away"],
                              fx["home"], fx["away"])
                    + f'<details><summary>Reasoning</summary><p>{expl}</p>'
                      f'</details></div>')
        groups_body += (f"<h3>Group {g}</h3><table><tr><th>Team</th>"
                        f"<th class='num'>Elo</th><th class='num'>Win group</th>"
                        f"<th class='num'>Reach R32</th><th class='num'>QF</th>"
                        f"<th class='num'>Champion</th></tr>{trows}</table>{fxs}")

    # ---------- methodology ----------
    bt_rows = "".join(
        f"<tr><td>{p['model']}</td><td class='num'>{p['logloss']:.4f}</td>"
        f"<td class='num'>{p['brier']:.4f}</td><td class='num'>{p['rps']:.4f}</td></tr>"
        for p in backtest["pooled"])
    meth_body = f"""
<h2>How this works</h2>
<p>Every prediction comes from a pipeline that is fully reproducible from public
data:</p>
<ol>
<li><b>Ratings.</b> We replay all {len(results):,} international matches since
1872 through a replication of the World Football Elo system (match-importance
K-factors, margin-of-victory multipliers, +100 home advantage). Our ratings
correlate {opening['elo_correlation_official']:.3f} with the official
eloratings.net figures across the 48 finalists.</li>
<li><b>Goals model.</b> Elo difference maps to expected goals via a Poisson
model fit by maximum likelihood on {forecast['params']['n']:,} competitive
internationals since 1998.</li>
<li><b>Tournament simulation.</b> We simulate the full 2026 format
{forecast['n_sims']:,} times — 12 groups with FIFA's 2026 tiebreakers
(head-to-head first, FIFA-ranking last), the eight best third-placed teams into
the new Round of 32, extra time at 0.33× expected goals, penalties as a coin
flip.</li>
<li><b>Market benchmark.</b> A bookmaker-consensus forecast (constant-overround
removal per book, geometric-mean pooling) from {consensus['n_outright_books']}
books on the outright market and up to 49 books per match.</li>
</ol>
<h2>Does it actually predict? The backtest</h2>
<p>The exact pipeline was replayed against the five World Cups 2006–2022 using
only information available before each tournament's opening day (ratings and
goals model refit per edition). Scores below are means over all 320 matches —
lower is better. The model beats both baselines on log loss; the gap to
bookmaker-grade accuracy is the roadmap.</p>
<table><tr><th>Model</th><th class="num">Log loss</th><th class="num">Brier</th>
<th class="num">RPS</th></tr>{bt_rows}</table>
<h2>Honest caveats</h2>
<ul>
<li>Elo sees results only — no squad values, injuries, or lineup news. That is
the main reason the model diverges from the market on teams like Argentina
(model higher) and Portugal (market higher).</li>
<li>Third-place bracket slots use constraint-satisfying assignment, not FIFA's
exact 495-combination allocation table.</li>
<li>The conduct-score (cards) tiebreaker is not simulated; the FIFA-ranking
tiebreaker is proxied by Elo.</li>
<li>No published model has ever been validated on a 48-team World Cup —
including this one. 2026 is new territory for everybody.</li>
</ul>
<p class="muted">Sources: martj42 international results dataset (CC0-style
community dataset), eloratings.net methodology, The Odds API snapshots,
academic recipes from Groll et al. (JQAS 2019) and Leitner/Zeileis/Hornik
(IJF 2010).</p>"""

    DOCS.mkdir(exist_ok=True)
    (DOCS / "style.css").write_text(CSS)
    (DOCS / "index.html").write_text(page("Forecast", index_body, as_of))
    (DOCS / "groups.html").write_text(page("Groups", groups_body, as_of))
    (DOCS / "methodology.html").write_text(page("Methodology", meth_body, as_of))
    print(f"Wrote docs/: index.html, groups.html, methodology.html, style.css")


if __name__ == "__main__":
    main()
