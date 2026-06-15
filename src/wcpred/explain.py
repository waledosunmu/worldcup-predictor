"""Templated, deterministic per-prediction explanations.

Every sentence is generated from model inputs/outputs and the raw results
data — no free-form generation — so published reasoning is always traceable
to a factor: Elo gap, expected goals, head-to-head record, recent form,
model-vs-market divergence.
"""

import pandas as pd

DIVERGENCE_PP = 0.07


def h2h_record(results: pd.DataFrame, a: str, b: str, before: str) -> dict:
    df = results[(results.date < before)
                 & (((results.home_team == a) & (results.away_team == b))
                    | ((results.home_team == b) & (results.away_team == a)))]
    df = df.dropna(subset=["home_score"])
    w = d = l = 0
    last = None
    for _, r in df.iterrows():
        ga = r.home_score if r.home_team == a else r.away_score
        gb = r.away_score if r.home_team == a else r.home_score
        w, d, l = w + (ga > gb), d + (ga == gb), l + (ga < gb)
        last = r
    rec = {"n": len(df), "w": int(w), "d": int(d), "l": int(l)}
    if last is not None:
        rec["last"] = (f"{last.date[:4]} {last.tournament}: "
                       f"{last.home_team} {int(last.home_score)}–"
                       f"{int(last.away_score)} {last.away_team}")
    return rec


def recent_form(results: pd.DataFrame, team: str, before: str, n: int = 8) -> dict:
    df = results[(results.date < before)
                 & ((results.home_team == team) | (results.away_team == team))]
    df = df.dropna(subset=["home_score"]).sort_values("date").tail(n)
    letters, gf, ga = [], 0, 0
    for _, r in df.iterrows():
        mine = r.home_score if r.home_team == team else r.away_score
        theirs = r.away_score if r.home_team == team else r.home_score
        gf, ga = gf + int(mine), ga + int(theirs)
        letters.append("W" if mine > theirs else ("D" if mine == theirs else "L"))
    return {"string": "".join(letters), "gf": gf, "ga": ga, "n": len(letters)}


def fixture_explanation(fx: dict, ratings: dict, ranks: dict,
                        results: pd.DataFrame, as_of: str,
                        market: dict | None = None) -> str:
    """One markdown paragraph of reasoning for a group fixture."""
    h, a = fx["home"], fx["away"]
    gap = round(ratings[h] - ratings[a])
    s = [f"Elo: {h} {ratings[h]:.0f} (#{ranks[h]}) vs {a} {ratings[a]:.0f} "
         f"(#{ranks[a]}) — a {abs(gap)}-point edge for {h if gap >= 0 else a}."]
    s.append(f"The goals model expects {fx['xg_home']}–{fx['xg_away']}, giving "
             f"{h} {fx['p_home']:.0%} / draw {fx['p_draw']:.0%} / "
             f"{a} {fx['p_away']:.0%}.")
    if market:
        s.append(f"Bookmaker consensus ({market['n_books']} books): "
                 f"{market['p_home']:.0%} / {market['p_draw']:.0%} / "
                 f"{market['p_away']:.0%}.")
        diff = fx["p_home"] - market["p_home"]
        if abs(diff) >= DIVERGENCE_PP:
            who = "more" if diff > 0 else "less"
            s.append(f"The model is notably {who} confident in {h} than the "
                     f"market ({diff:+.0%}) — Elo ratings reflect results only, "
                     f"not squad value, injuries or lineup news.")
    rec = h2h_record(results, h, a, as_of)
    if rec["n"]:
        s.append(f"Head-to-head: {rec['w']}W {rec['d']}D {rec['l']}L for {h} "
                 f"in {rec['n']} meetings (last: {rec['last']}).")
    else:
        s.append("These sides have never met in a full international.")
    for t in (h, a):
        f = recent_form(results, t, as_of)
        if f["n"]:
            s.append(f"{t} form (last {f['n']}): {f['string']}, "
                     f"{f['gf']}–{f['ga']} goals.")
    return " ".join(s)


def team_blurb(team: str, adv: dict, rating: float, rank: int, group: str,
               market_champ: float | None = None) -> str:
    s = [f"Elo {rating:.0f} (#{rank} of the field), Group {group}.",
         f"Simulation: {adv['reach_r32']:.0%} to reach the R32, "
         f"{adv['win_group']:.0%} to win the group, "
         f"{adv['reach_sf']:.0%} semi-final, {adv['champion']:.1%} champion."]
    if market_champ is not None:
        diff = adv["champion"] - market_champ
        if abs(diff) >= 0.03:
            view = "higher on" if diff > 0 else "lower on"
            s.append(f"The model is {view} {team} than the betting market "
                     f"({adv['champion']:.1%} vs {market_champ:.1%}).")
    return " ".join(s)


_SIDES = ("home", "draw", "away")


def pick_from(probs: dict) -> tuple[str, float]:
    """Modal W/D/L pick (side, probability) from a {p_home,p_draw,p_away} dict."""
    vals = [probs["p_home"], probs["p_draw"], probs["p_away"]]
    i = max(range(3), key=lambda k: vals[k])
    return _SIDES[i], vals[i]


def result_review(home: str, away: str, sh: int, sa: int, outcome: str,
                  model: dict, market: dict | None = None) -> str:
    """Post-match narrative: how the pre-kickoff model (and market) call fared.

    `model`/`market` are {p_home,p_draw,p_away} locked before kickoff; `outcome`
    is the actual "home"/"draw"/"away".
    """
    def label(side):
        return {"home": home, "away": away, "draw": "a draw"}[side]

    winner = home if sh > sa else away if sa > sh else None
    s = [f"{home} {sh}–{sa} {away} — "
         + (f"{winner} won." if winner else "a draw.")]

    pick, prob = pick_from(model)
    if pick == outcome:
        s.append(f"The model called it: it leaned {label(pick)} ({prob:.0%}).")
    else:
        actual_p = [model["p_home"], model["p_draw"], model["p_away"]][_SIDES.index(outcome)]
        s.append(f"The model missed — it favoured {label(pick)} ({prob:.0%}) and "
                 f"gave the actual result only {actual_p:.0%}.")

    if market:
        mpick, mprob = pick_from(market)
        mhit = mpick == outcome
        if mpick == pick:
            s.append(f"The market agreed ({label(mpick)} {mprob:.0%}) and "
                     f"{'also called it' if mhit else 'also missed'}.")
        else:
            s.append(f"The market instead leaned {label(mpick)} ({mprob:.0%}) — "
                     f"{'which landed' if mhit else 'also wrong'}.")
    return " ".join(s)
