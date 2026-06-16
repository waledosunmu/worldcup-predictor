"""Monte Carlo simulation of the 2026 World Cup (48 teams, 12 groups, R32).

Follows the Groll et al. recipe: Poisson scores from expected goals, official
group rules, extra time as 0.33x expected goals, penalties as a coin flip.

2026-specific rules encoded from FIFA via data/format_2026.json:
  - top 2 per group + 8 best third-placed teams reach the Round of 32
  - group tiebreakers: points; head-to-head (points/GD/goals) among tied teams;
    overall GD; overall goals; [conduct score - not simulated]; FIFA ranking
    (proxied here by pre-tournament Elo)
  - third-place table: points, GD, goals, [conduct], FIFA-ranking proxy
  - third-place bracket slots: constraint-satisfying assignment over the slot
    eligibility sets (approximation of FIFA's 495-combination table)

Hosts (Mexico, Canada, United States) get the +100 Elo home advantage in the
group stage, where they are guaranteed to play in their own country; knockout
matches are treated as neutral for everyone.
"""

import itertools
import json

import numpy as np

from . import covariates, dixoncoles
from .goals import expected_goals

HOSTS = {"Mexico", "Canada", "United States"}
ET_FACTOR = 0.33
THIRD_SLOTS = ["74", "77", "79", "80", "81", "82", "85", "87"]


def load_format(path):
    with open(path) as f:
        return json.load(f)


class Simulator:
    """`played_group`: {frozenset({a,b}): (a, goals_a, goals_b)} fixes group
    scores; `ko_winners`: {frozenset({a,b}): winner} fixes knockout outcomes
    (shootout winners included). Everything not fixed is simulated."""

    def __init__(self, fmt: dict, ratings: dict[str, float], params: dict, seed: int = 26,
                 played_group: dict | None = None, ko_winners: dict | None = None,
                 dixon_coles: bool = False, team_form: dict[str, float] | None = None):
        self.fmt = fmt
        self.params = params
        self.ratings = ratings
        # When dixon_coles=True, scorelines are drawn from the DC-corrected joint
        # grid (params must carry 'rho'); otherwise the v0 independent-Poisson
        # path is used unchanged (default keeps the live pipeline behaviour).
        self.dixon_coles = dixon_coles
        # team_form maps team -> recent-form value; when params carry covariate
        # betas (the hybrid), expected goals are covariate-augmented with the
        # fixture's form_diff. Absent/None -> the v0 Elo-only means, unchanged.
        self.team_form = team_form
        self.played_group = played_group or {}
        self.ko_winners = ko_winners or {}
        self.rng = np.random.default_rng(seed)
        self.group_letters = sorted(fmt["groups"])
        self.groups = {g: list(fmt["groups"][g]) for g in self.group_letters}
        self.teams = [t for g in self.group_letters for t in self.groups[g]]
        # third-place slot eligibility from the bracket spec ("3:ABCDF")
        self.third_eligible = {
            m: set(spec["away"].split(":")[1])
            for m, spec in fmt["round_of_32"].items() if spec["away"].startswith("3:")
        }
        # fixed group fixtures: round robin, hosts at home
        self.group_fixtures = []
        for g in self.group_letters:
            for a, b in itertools.combinations(self.groups[g], 2):
                home, away = (a, b) if (b not in HOSTS) else (b, a)
                lh, la = self._means(home, away, 100.0 if home in HOSTS else 0.0)
                self.group_fixtures.append((g, home, away, lh, la))

    def _means(self, home: str, away: str, host_adv: float) -> tuple[float, float]:
        """Expected (home, away) goals — covariate-augmented when the hybrid
        params + team_form are present, else the v0 Elo-only means."""
        dr = self.ratings[home] - self.ratings[away] + host_adv
        if self.params.get("beta") and self.team_form is not None:
            z = {"form_diff": self.team_form.get(home, 0.0) - self.team_form.get(away, 0.0)}
            return covariates.expected_goals(dr, z, self.params)
        return expected_goals(dr, self.params)

    # ---------- match simulation ----------

    def _ko_winner(self, a: str, b: str) -> str:
        fixed = self.ko_winners.get(frozenset((a, b)))
        if fixed is not None:
            return fixed
        lh, la = self._means(a, b, 0.0)  # knockouts neutral for everyone
        ga, gb = self._draw_score(lh, la)
        if ga != gb:
            return a if ga > gb else b
        ga, gb = self._draw_score(lh * ET_FACTOR, la * ET_FACTOR)
        if ga != gb:
            return a if ga > gb else b
        return a if self.rng.random() < 0.5 else b

    def _draw_score(self, lh: float, la: float) -> tuple[int, int]:
        """One (home, away) scoreline — DC grid draw if enabled, else Poisson."""
        if self.dixon_coles:
            s = dixoncoles.sample_scores(lh, la, self.params["rho"], self.rng, 1)[0]
            return int(s[0]), int(s[1])
        return int(self.rng.poisson(lh)), int(self.rng.poisson(la))

    # ---------- group stage ----------

    def _rank_group(self, g: str, stats: dict, h2h: dict) -> list[str]:
        """stats: team -> [pts, gf, ga]; h2h: (x,y) -> (gx, gy)."""
        teams = self.groups[g]

        def overall_key(t):
            pts, gf, ga = stats[t]
            return (pts, gf - ga, gf, self.ratings[t])

        ordered = sorted(teams, key=overall_key, reverse=True)
        # re-rank clusters tied on points by head-to-head among the tied subset
        final: list[str] = []
        i = 0
        while i < len(ordered):
            cluster = [t for t in ordered if stats[t][0] == stats[ordered[i]][0]]
            if len(cluster) > 1:
                mini = {t: [0, 0, 0] for t in cluster}
                for x, y in itertools.combinations(cluster, 2):
                    gx, gy = h2h[(x, y)] if (x, y) in h2h else h2h[(y, x)][::-1]
                    mini[x][1] += gx; mini[x][2] += gy
                    mini[y][1] += gy; mini[y][2] += gx
                    if gx > gy:
                        mini[x][0] += 3
                    elif gy > gx:
                        mini[y][0] += 3
                    else:
                        mini[x][0] += 1; mini[y][0] += 1

                def cluster_key(t):
                    hp, hgf, hga = mini[t]
                    pts, gf, ga = stats[t]
                    return (hp, hgf - hga, hgf, gf - ga, gf, self.ratings[t])

                cluster.sort(key=cluster_key, reverse=True)
            final.extend(cluster)
            i += len(cluster)
        return final

    def _allocate_thirds(self, qualified: list[tuple[str, str]]) -> dict[str, str]:
        """qualified: [(group_letter, team)] for the 8 best thirds.
        Returns slot -> team via most-constrained-first backtracking."""
        letters = {g: t for g, t in qualified}
        slots = sorted(THIRD_SLOTS, key=lambda m: len(self.third_eligible[m] & letters.keys()))

        assignment: dict[str, str] = {}
        used: set[str] = set()

        def backtrack(idx: int) -> bool:
            if idx == len(slots):
                return True
            slot = slots[idx]
            # sorted(): set iteration order over group-letter strings is
            # hash-randomized per process, so an unsorted scan would pick a
            # different (still valid) third-place allocation each run and make
            # the knockout probabilities non-reproducible (cardinal rule #2).
            for g in sorted(self.third_eligible[slot] & letters.keys()):
                if g not in used:
                    used.add(g)
                    assignment[slot] = letters[g]
                    if backtrack(idx + 1):
                        return True
                    used.discard(g)
                    del assignment[slot]
            return False

        if not backtrack(0):  # should not happen per FIFA's table; degrade gracefully
            assignment = dict(zip(THIRD_SLOTS, [t for _, t in qualified]))
        return assignment

    # ---------- one tournament ----------

    def simulate_once(self, group_goals: np.ndarray) -> dict:
        """group_goals: (72, 2) pre-sampled scores for self.group_fixtures."""
        stats = {t: [0, 0, 0] for t in self.teams}
        h2h: dict[tuple[str, str], tuple[int, int]] = {}
        for (g, home, away, _, _), (gh, ga) in zip(self.group_fixtures, group_goals):
            gh, ga = int(gh), int(ga)
            h2h[(home, away)] = (gh, ga)
            stats[home][1] += gh; stats[home][2] += ga
            stats[away][1] += ga; stats[away][2] += gh
            if gh > ga:
                stats[home][0] += 3
            elif ga > gh:
                stats[away][0] += 3
            else:
                stats[home][0] += 1; stats[away][0] += 1

        ranks = {g: self._rank_group(g, stats, h2h) for g in self.group_letters}
        slots: dict[str, str] = {}
        for g, order in ranks.items():
            slots[f"1{g}"] = order[0]
            slots[f"2{g}"] = order[1]

        thirds = [(g, ranks[g][2]) for g in self.group_letters]

        def third_key(item):
            g, t = item
            pts, gf, ga = stats[t]
            return (pts, gf - ga, gf, self.ratings[t])

        thirds.sort(key=third_key, reverse=True)
        qualified_thirds = thirds[:8]
        third_assignment = self._allocate_thirds(qualified_thirds)

        winners: dict[str, str] = {}
        r32_pairs: dict[str, tuple[str, str]] = {}
        for m, spec in self.fmt["round_of_32"].items():
            home = slots[spec["home"]]
            away = third_assignment[m] if spec["away"].startswith("3:") else slots[spec["away"]]
            r32_pairs[m] = (home, away)
            winners[m] = self._ko_winner(home, away)

        rounds = {"r16": {}, "qf": {}, "sf": {}, "final": {}}
        for key, stage in (("round_of_16", "r16"), ("quarterfinals", "qf"),
                           ("semifinals", "sf"), ("final", "final")):
            for m, (m1, m2) in self.fmt[key].items():
                a, b = winners[m1], winners[m2]
                rounds[stage][m] = (a, b)
                winners[m] = self._ko_winner(a, b)

        return {
            "ranks": ranks,
            "qualified_thirds": [t for _, t in qualified_thirds],
            "r32": r32_pairs,
            "rounds": rounds,
            "winners": winners,
            "champion": winners["104"],
        }

    # ---------- aggregation ----------

    def run(self, n_sims: int = 100_000) -> dict:
        nf = len(self.group_fixtures)
        if self.dixon_coles:
            all_goals = np.empty((n_sims, nf, 2), dtype=int)
            for idx, f in enumerate(self.group_fixtures):
                all_goals[:, idx, :] = dixoncoles.sample_scores(
                    f[3], f[4], self.params["rho"], self.rng, n_sims)
        else:
            lam = np.array([[f[3], f[4]] for f in self.group_fixtures])  # (nf, 2)
            all_goals = self.rng.poisson(lam, size=(n_sims, nf, 2))
        # overwrite played fixtures with their actual scores in every simulation
        for idx, (g, home, away, _, _) in enumerate(self.group_fixtures):
            rec = self.played_group.get(frozenset((home, away)))
            if rec is not None:
                first, gf, ga = rec
                score = (gf, ga) if first == home else (ga, gf)
                all_goals[:, idx, :] = score

        stages = ["win_group", "reach_r32", "reach_r16", "reach_qf",
                  "reach_sf", "reach_final", "champion"]
        counts = {t: dict.fromkeys(stages, 0) for t in self.teams}

        for i in range(n_sims):
            sim = self.simulate_once(all_goals[i])
            for g, order in sim["ranks"].items():
                counts[order[0]]["win_group"] += 1
                counts[order[0]]["reach_r32"] += 1
                counts[order[1]]["reach_r32"] += 1
            for t in sim["qualified_thirds"]:
                counts[t]["reach_r32"] += 1
            for m, (a, b) in sim["rounds"]["r16"].items():
                counts[a]["reach_r16"] += 1; counts[b]["reach_r16"] += 1
            for m, (a, b) in sim["rounds"]["qf"].items():
                counts[a]["reach_qf"] += 1; counts[b]["reach_qf"] += 1
            for m, (a, b) in sim["rounds"]["sf"].items():
                counts[a]["reach_sf"] += 1; counts[b]["reach_sf"] += 1
            a, b = sim["rounds"]["final"]["104"]
            counts[a]["reach_final"] += 1; counts[b]["reach_final"] += 1
            counts[sim["champion"]]["champion"] += 1

        probs = {
            t: {s: c / n_sims for s, c in cs.items()} for t, cs in counts.items()
        }
        return {"n_sims": n_sims, "probs": probs}
