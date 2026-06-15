"""Render the prediction site to docs/ (GitHub Pages-ready) from model artifacts.

Inputs: output/forecast_opening.json, output/consensus.json,
        output/backtest_2006_2022.json, data/format_2026.json, data/raw/results.csv
Usage:  python scripts/build_site.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpred import metrics
from wcpred.explain import fixture_explanation, team_blurb
from wcpred.history import load_jsonl

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

/* ---- interactive enhancements (app.js) ---- */
.controls { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center;
            margin:.8rem 0; }
.controls input[type=text], .controls select { padding:.35rem .5rem;
            border:1px solid var(--line); border-radius:6px; font-size:.9rem;
            background:var(--card); color:var(--ink); }
.controls .seg { display:inline-flex; border:1px solid var(--line);
            border-radius:6px; overflow:hidden; }
.controls .seg button { border:0; background:var(--card); color:var(--muted);
            padding:.35rem .7rem; cursor:pointer; font-size:.85rem; font-weight:600; }
.controls .seg button.active { background:var(--model); color:#fff; }
.controls .seg button.active.market { background:var(--market); }
th.sortable { cursor:pointer; user-select:none; }
th.sortable::after { content:" \\2195"; color:var(--muted); font-size:.8em; }
th.sortable.asc::after { content:" \\2191"; color:var(--ink); }
th.sortable.desc::after { content:" \\2193"; color:var(--ink); }
a.team-link { color:var(--model); text-decoration:none; font-weight:650;
            cursor:pointer; }
a.team-link:hover { text-decoration:underline; }
.chartwrap { background:var(--card); border:1px solid var(--line);
            border-radius:8px; padding:1rem; margin:.8rem 0; }
.chartwrap canvas { max-width:100%; }
.checks { display:flex; flex-wrap:wrap; gap:.5rem 1rem; margin:.6rem 0; }
.checks label { font-size:.85rem; display:inline-flex; align-items:center;
            gap:.3rem; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
            gap:.8rem; margin:.8rem 0; }
.mcard { background:var(--card); border:1px solid var(--line); border-radius:8px;
            padding:.8rem 1rem; }
.mcard .delta-up { color:#16a34a; font-weight:650; }
.mcard .delta-down { color:#dc2626; font-weight:650; }
.badge { display:inline-block; font-size:.72rem; font-weight:700; padding:.1rem .45rem;
            border-radius:10px; color:#fff; }
.badge.model { background:var(--model); } .badge.market { background:var(--market); }
.scorebadge { font-weight:700; color:var(--ink); }
/* bracket */
#bracket { overflow-x:auto; padding-bottom:1rem; }
.bracket-cols { display:flex; gap:1rem; min-width:760px; }
.bcol { flex:1; min-width:120px; }
.bcol h4 { margin:.2rem 0 .5rem; font-size:.82rem; text-transform:uppercase;
            letter-spacing:.04em; color:var(--muted); }
.bteam { font-size:.82rem; padding:.3rem .5rem; margin:.25rem 0; border-radius:5px;
            border:1px solid var(--line); cursor:pointer; display:flex;
            justify-content:space-between; gap:.4rem; }
.bteam b { font-weight:600; }
.bteam span.p { color:inherit; opacity:.72; font-variant-numeric:tabular-nums; }
/* modal */
#team-modal { position:fixed; inset:0; background:rgba(15,23,42,.55);
            display:none; align-items:flex-start; justify-content:center;
            padding:2rem 1rem; z-index:50; overflow-y:auto; }
#team-modal.open { display:flex; }
#team-modal .box { background:var(--card); border-radius:12px; max-width:640px;
            width:100%; padding:1.3rem 1.5rem; box-shadow:0 20px 60px rgba(0,0,0,.3); }
#team-modal h3 { margin:.1rem 0; border:0; }
#team-modal .close { float:right; cursor:pointer; border:0; background:none;
            font-size:1.4rem; color:var(--muted); line-height:1; }
.advrow { display:flex; align-items:center; gap:.6rem; margin:.3rem 0;
            font-size:.85rem; }
.advrow .lbl { width:90px; color:var(--muted); }
.advrow .pct { width:46px; text-align:right; font-variant-numeric:tabular-nums; }
.advrow .bar { flex:1; }
.modal-sched .fixture { margin:.5rem 0; }
.muted.small { font-size:.78rem; margin:.5rem 0 .2rem; }
/* live track-record per-match table */
.tr-matches { font-size:.82rem; }
.tr-matches th, .tr-matches td { padding:.3rem .4rem; }
.tr-matches .mtch { white-space:nowrap; }
.tr-matches .vs { color:var(--muted); }
.tr-matches .sc { color:var(--muted); font-variant-numeric:tabular-nums; }
.tr-matches td.win { color:#15803d; font-weight:650; }
/* ---------- mobile ---------- */
@media (max-width:640px) {
  header { padding:1.1rem .9rem; }
  header h1 { font-size:1.3rem; }
  header p { font-size:.9rem; }
  nav a { margin-right:.9rem; }
  main { padding:1rem .8rem 3rem; }
  h2 { margin-top:1.6rem; font-size:1.25rem; }
  /* wide tables scroll horizontally instead of breaking the layout */
  main > table, .tablewrap, #team-table, .tr-matches,
  .groups-table, table { display:block; max-width:100%; overflow-x:auto;
            white-space:nowrap; -webkit-overflow-scrolling:touch; }
  .bar { min-width:70px; }
  #team-modal { padding:0; align-items:stretch; }
  #team-modal .box { max-width:100%; min-height:100%; border-radius:0;
            padding:1.1rem 1rem; }
}
"""


def load_optional(path: Path):
    return json.load(open(path)) if path.exists() else None


def page(title: str, body: str, updated: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — WC26 Predictor</title><link rel="stylesheet" href="style.css"></head>
<body><header><div class="inner"><h1>⚽ World Cup 2026 Predictor</h1>
<p>Probabilistic forecasts with the reasoning behind every prediction.</p>
<nav><a href="index.html">Forecast</a><a href="groups.html">Groups &amp; matches</a>
<a href="trends.html">Trends</a><a href="bracket.html">Bracket</a>
<a href="methodology.html">Methodology</a></nav></div></header>
<main>{body}<footer>Updated {updated}. Model: Elo–Poisson v0, 100k Monte Carlo
simulations. Market: bookmaker consensus via The Odds API. Data: martj42
international results, eloratings.net. Probabilities are estimates, not advice.
</footer></main>
<div id="team-modal" aria-hidden="true"><div class="box"><button class="close"
aria-label="Close">&times;</button><div id="team-modal-content"></div></div></div>
<script src="vendor/chart.min.js"></script>
<script src="app.js" defer></script>
</body></html>"""


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
    tbt = load_optional(ROOT / "output/tournament_backtest_2006_2022.json")
    mc = load_optional(ROOT / "output/model_comparison.json")
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
    rows = sorted(adv.items(), key=lambda kv: -kv[1]["champion"])
    champ_rows = ""
    for t, p in rows:
        mk = market_champ.get(t, 0.0)
        champ_rows += (
            f"<tr data-group='{team_group[t]}'>"
            f"<td><a class='team-link' data-team=\"{t}\">{t}</a></td>"
            f"<td>{team_group[t]}</td>"
            f"<td class='num' data-v='{ratings[t]:.0f}'>{ratings[t]:.0f}</td>"
            f"<td class='num' data-v='{p['champion']}'>{p['champion']:.1%}"
            f"{bar(p['champion'], 'model', max_champ)}</td>"
            f"<td class='num' data-v='{mk}'>{mk:.1%}"
            f"{bar(mk, 'market', max_champ)}</td>"
            f"<td class='num' data-v='{opening_champ.get(t, 0)}'>{opening_champ.get(t, 0):.1%}</td>"
            f"<td class='num' data-v='{p['reach_final']}'>{p['reach_final']:.0%}</td>"
            f"<td class='num' data-v='{p['reach_sf']}'>{p['reach_sf']:.0%}</td></tr>")

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
<div class="controls" id="table-controls">
<input type="text" id="team-search" placeholder="Search team…" aria-label="Search team">
<select id="group-filter" aria-label="Filter by group"><option value="">All groups</option>
{"".join(f"<option value='{g}'>Group {g}</option>" for g in sorted(fmt['groups']))}</select>
<span class="seg" id="champ-toggle"><button data-mode="model" class="active">Model</button>
<button data-mode="market">Market</button></span></div>
<table id="team-table"><thead><tr>
<th class="sortable" data-key="team">Team</th>
<th class="sortable" data-key="group">Group</th>
<th class="sortable num" data-key="elo">Elo</th>
<th class="sortable num" data-key="model_champion" data-default-desc="1">Champion (model)</th>
<th class="sortable num" data-key="market_champion">Champion (market)</th>
<th class="sortable num" data-key="pre">Pre-tournament</th>
<th class="sortable num" data-key="reach_final">Final</th>
<th class="sortable num" data-key="reach_sf">Semis</th></tr></thead>
<tbody>{champ_rows}</tbody></table>
<h2>Where the model disagrees with the market</h2>
<p class="muted">Latest model champion probability minus bookmaker consensus, per team.
Positive = model is higher. Hover a bar for the team's profile.</p>
<div class="chartwrap"><div id="divergence"></div></div>
<h2>Title-odds movers &amp; live track record</h2>
<div class="cards"><div id="movers"></div><div id="track-record"></div></div>
<h2>Next matches</h2>{up_html}
<p><a href="groups.html">All 72 group fixtures with predictions →</a></p>"""

    # ---------- groups ----------
    # Kickoff date per fixture (covers played + scheduled) from the results
    # snapshot, so fixtures list in the order they're played, not alphabetically.
    wc_rows = results[(results.tournament == "FIFA World Cup")
                      & (results.date >= "2026-06-01")]
    fixture_date = {frozenset((r.home_team, r.away_team)): r.date
                    for _, r in wc_rows.iterrows()}

    def fixture_order(fx):
        d = fixture_date.get(frozenset((fx["home"], fx["away"])), "9999-99-99")
        m = market_by_fixture.get((fx["home"], fx["away"]))
        return (d, m["commence"] if m else "")

    fixtures_by_group = {}
    for fx in forecast["group_fixtures"]:
        fixtures_by_group.setdefault(fx["group"], []).append(fx)
    for g in fixtures_by_group:
        fixtures_by_group[g].sort(key=fixture_order)
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
            date = (m["commence"][:10] if m
                    else fixture_date.get(frozenset((fx["home"], fx["away"])), ""))
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

    # ---------- whole-tournament backtest (optional artifact) ----------
    tbt_section = ""
    if tbt:
        s = tbt["summary"]
        flat = 1 / 32
        tt_rows = ""
        for e in tbt["per_edition"]:
            ep = e["elo_poisson"]
            pick, pick_p = ep["top5_champion"][0]
            hit = "✓" if pick == e["actual_champion"] else ""
            tt_rows += (
                f"<tr><td class='num'>{e['edition']}</td>"
                f"<td>{e['actual_champion']}</td>"
                f"<td>{pick} {pick_p:.0%} {hit}</td>"
                f"<td class='num'>{ep['p_actual_champion']:.1%}"
                f"{bar(ep['p_actual_champion'], 'model', 0.35)}</td>"
                f"<td class='num'>{ep['p_actual_finalists']:.1%}</td></tr>")
        champ = s["champion"]
        tbt_section = f"""
<h2>Whole-tournament backtest: does it call the champion?</h2>
<p>The match-level scores above ask “how good is each prediction?” This asks the
harder question: simulate each past World Cup end to end (refit before kickoff)
and see how much probability the model put on the team that actually lifted the
trophy. A blind guess is 1/32 = {flat:.1%}.</p>
<table><tr><th class="num">Year</th><th>Champion</th>
<th>Model's top pick</th><th class="num">P(actual champ)</th>
<th class="num">P(reach final)</th></tr>{tt_rows}</table>
<p class="muted">Across all five editions the model gave the eventual champion
<b>{champ['elo_poisson_mean_p_actual']:.1%}</b> on average — {champ['elo_poisson_mean_p_actual'] / flat:.1f}× a
random pick — and the eventual finalists
<b>{s['reach_final']['mean_p_actual_finalists']:.1%}</b>. Reach-final probabilities
are well calibrated: predicted bins track observed frequencies across
{sum(b['n'] for b in s['reach_final']['reliability'])} team-edition outcomes.
2006 (Italy) and 2018 (France) were genuine upsets no rating-based model saw
coming.</p>"""

    # ---------- model variants (optional artifact) ----------
    mc_section = ""
    if mc:
        labels = {
            "v0_poisson": "Elo–Poisson v0 (live default)",
            "dixon_coles": "+ Dixon-Coles low-score correction",
            "covar_form": "+ recent-form covariate",
            "hybrid_form_dc": "Hybrid: form + Dixon-Coles",
        }
        oos = mc["oos_pooled"]
        best = min(labels, key=lambda k: oos[k]["logloss"])
        mc_rows = ""
        for k, label in labels.items():
            o = oos[k]
            b = " style='font-weight:650'" if k == best else ""
            mc_rows += (
                f"<tr{b}><td>{label}</td><td class='num'>{o['logloss']:.4f}</td>"
                f"<td class='num'>{o['brier']:.4f}</td>"
                f"<td class='num'>{o['rps']:.4f}</td></tr>")
        mc_section = f"""
<h2>Model variants: what's on the roadmap</h2>
<p>Beyond v0, two research extensions are implemented and backtested
point-in-time over the same 320 matches (lower is better). The Dixon-Coles
correction adjusts low-scoring scorelines; the covariate model adds recent form
to the Elo signal. The combined hybrid wins on every metric — but the gains are
modest, so v0 remains the production default until they earn their place.</p>
<table><tr><th>Model</th><th class="num">Log loss</th><th class="num">Brier</th>
<th class="num">RPS</th></tr>{mc_rows}</table>
<p class="muted">Groll-style squad covariates (market value, squad age,
Champions-League minutes) are the next lever but need external per-tournament
squad data not yet wired in.</p>"""

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
<h2>Live 2026 track record</h2>
<p>As 2026 matches are played we lock the last pre-kickoff model and market
prediction for each fixture and score them out-of-sample. This panel fills in
match by match — it is the same numbers shown on the forecast page.</p>
<div id="track-record"></div>
{tbt_section}{mc_section}
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

    # ---------- trends ----------
    history = load_jsonl(ROOT / "output/probability_history.jsonl")
    trends_body = f"""
<h2>Championship trends over time</h2>
<p class="muted">How each team's title odds (and reach-final / Elo) have moved as
the tournament unfolds. The series is sparse early — there are
{len(history)} snapshot{"s" if len(history) != 1 else ""} so far; it grows one
point per matchday.</p>
<div class="controls">
<label for="trend-metric">Metric</label>
<select id="trend-metric"><option value="model_champion">Champion</option>
<option value="model_reach_final">Reach final</option>
<option value="elo">Elo</option></select>
<span class="seg" id="trend-toggle"><button data-mode="model" class="active">Model</button>
<button data-mode="market">Market</button></span></div>
<div class="checks" id="trend-teams"></div>
<div class="chartwrap"><canvas id="trend-chart" height="320"></canvas></div>
<p class="muted" id="trend-caption"></p>"""

    # ---------- bracket ----------
    bracket_body = """
<h2>Interactive knockout bracket</h2>
<p class="muted">Each round lists every team shaded by its simulated probability
of reaching that stage. Hover a round header for its likeliest occupants.
Scroll horizontally on mobile.</p>
<div id="bracket"></div>"""

    # ---------- data.json payload ----------
    teams_payload = []
    for t in sorted(adv, key=lambda x: -adv[x]["champion"]):
        p = adv[t]
        mk = market_champ.get(t, 0.0)
        teams_payload.append({
            "name": t, "group": team_group[t], "elo": round(ratings[t], 1),
            "rank": ranks[t], "model_champion": p["champion"], "market_champion": mk,
            "win_group": p["win_group"], "reach_r32": p["reach_r32"],
            "reach_r16": p["reach_r16"], "reach_qf": p["reach_qf"],
            "reach_sf": p["reach_sf"], "reach_final": p["reach_final"],
            "blurb": team_blurb(t, p, ratings[t], ranks[t], team_group[t], mk),
        })

    mp_by_pair = {(m["home"], m["away"]): m
                  for m in load_jsonl(ROOT / "output/match_predictions.jsonl")}
    matches_payload = []
    for fx in forecast["group_fixtures"]:
        pair = (fx["home"], fx["away"])
        mp = mp_by_pair.get(pair)
        mkt = market_by_fixture.get(pair)
        rec = {"home": fx["home"], "away": fx["away"], "group": fx["group"],
               "commence": mp["commence"] if mp else None,
               "date": (mp["commence"][:10] if mp and mp.get("commence") else None),
               "played": bool(fx.get("played"))}
        if fx.get("played"):
            rec.update({"score_home": fx.get("score_home"),
                        "score_away": fx.get("score_away"),
                        "model": None, "market": None, "explanation": None})
        else:
            market_mk = ({"p_home": mkt["p_home"], "p_draw": mkt["p_draw"],
                          "p_away": mkt["p_away"], "n_books": mkt["n_books"]}
                         if mkt else None)
            rec.update({
                "score_home": None, "score_away": None,
                "model": {"p_home": fx["p_home"], "p_draw": fx["p_draw"],
                          "p_away": fx["p_away"]},
                "market": market_mk,
                "explanation": fixture_explanation(fx, ratings, ranks, results,
                                                   as_of, mkt),
            })
        matches_payload.append(rec)

    history_payload = [{
        "as_of": h["as_of"], "n_played_group": h.get("n_played_group", 0),
        "model_champion": h["model"]["champion"],
        "model_reach_final": h["model"]["reach_final"],
        "market_champion": h["market"]["champion"],
        "elo": h["model"]["elo"],
    } for h in sorted(history, key=lambda h: h["as_of"])]

    divergence_payload = sorted(
        ({"team": t, "model": adv[t]["champion"],
          "market": market_champ.get(t, 0.0),
          "diff": adv[t]["champion"] - market_champ.get(t, 0.0)} for t in adv),
        key=lambda d: -abs(d["diff"]))

    movers_payload = []
    if len(history_payload) >= 2:
        prev, cur = history_payload[-2], history_payload[-1]
        deltas = []
        for t in cur["model_champion"]:
            frm = prev["model_champion"].get(t)
            if frm is None:
                continue
            to = cur["model_champion"][t]
            deltas.append({"team": t, "from": frm, "to": to, "delta": to - frm})
        deltas.sort(key=lambda d: d["delta"])
        fallers = [d for d in deltas if d["delta"] < 0][:5]
        risers = [d for d in reversed(deltas) if d["delta"] > 0][:5]
        movers_payload = risers + fallers

    # ---------- live track record ----------
    OUT_IDX = {"home": 0, "draw": 1, "away": 2}
    scored = [m for m in mp_by_pair.values()
              if m.get("played") and m.get("model") and m.get("market")
              and m.get("result")]
    track_record = None
    if scored:
        agg = {"model": {"logloss": 0.0, "brier": 0.0, "rps": 0.0},
               "market": {"logloss": 0.0, "brier": 0.0, "rps": 0.0}}
        tr_matches = []
        for m in scored:
            oi = OUT_IDX[m["result"]["outcome"]]
            row = {"home": m["home"], "away": m["away"],
                   "outcome": m["result"]["outcome"],
                   "score": [m["result"]["score_home"], m["result"]["score_away"]]}
            for side in ("model", "market"):
                pr = np.array([m[side]["p_home"], m[side]["p_draw"],
                               m[side]["p_away"]], dtype=float)
                ll = metrics.log_loss(pr, oi)
                br = metrics.brier(pr, oi)
                rp = metrics.rps(pr, oi)
                agg[side]["logloss"] += ll
                agg[side]["brier"] += br
                agg[side]["rps"] += rp
                row[side] = {"p_home": m[side]["p_home"],
                             "p_draw": m[side]["p_draw"],
                             "p_away": m[side]["p_away"],
                             "logloss": ll, "brier": br, "rps": rp}
            tr_matches.append(row)
        n = len(scored)
        for side in ("model", "market"):
            for k in agg[side]:
                agg[side][k] /= n
        track_record = {"model": agg["model"], "market": agg["market"],
                        "n": n, "matches": tr_matches}

    data_payload = {
        "meta": {
            "as_of": as_of, "generated": forecast.get("generated"),
            "n_played_group": forecast.get("n_played_group", 0),
            "n_played_ko": forecast.get("n_played_ko", 0),
            "n_outright_books": consensus["n_outright_books"],
            "n_history_points": len(history_payload),
        },
        "teams": teams_payload,
        "matches": matches_payload,
        "history": history_payload,
        "divergence": divergence_payload,
        "movers": movers_payload,
        "trackRecord": track_record,
    }

    DOCS.mkdir(exist_ok=True)
    (DOCS / "style.css").write_text(CSS)
    (DOCS / "index.html").write_text(page("Forecast", index_body, as_of))
    (DOCS / "groups.html").write_text(page("Groups", groups_body, as_of))
    (DOCS / "methodology.html").write_text(page("Methodology", meth_body, as_of))
    (DOCS / "trends.html").write_text(page("Trends", trends_body, as_of))
    (DOCS / "bracket.html").write_text(page("Bracket", bracket_body, as_of))
    (DOCS / "data.json").write_text(json.dumps(data_payload))
    print("Wrote docs/: index, groups, methodology, trends, bracket .html, "
          "data.json, style.css")


if __name__ == "__main__":
    main()
