"""Glicko ratings for tennis.

Glicko improves on Elo by tracking each player's rating *deviation* (RD) — how
uncertain we are about their strength. That uncertainty feeds directly into a
wider probability band for lightly-played or long-inactive players, which is
exactly what we want for an honest value tool.

This is the Glicko-1 formulation (Glickman, 1999), in pure Python.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .base import InsufficientModelData, ModelOutput

_Q = math.log(10) / 400.0  # ln(10)/400


@dataclass
class GlickoRating:
    rating: float = 1500.0
    rd: float = 350.0  # rating deviation; high = uncertain
    games: int = 0


def _g(rd: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * _Q * _Q * rd * rd / (math.pi * math.pi))


def _expected(r: float, r_j: float, rd_j: float) -> float:
    return 1.0 / (1.0 + 10 ** (-_g(rd_j) * (r - r_j) / 400.0))


@dataclass
class GlickoModel:
    c: float = 34.0  # RD inflation per period of inactivity
    base_rd: float = 350.0
    ratings: dict[str, GlickoRating] = field(default_factory=dict)

    def _get(self, player: str) -> GlickoRating:
        return self.ratings.setdefault(player, GlickoRating(rd=self.base_rd))

    def update(self, winner: str, loser: str) -> None:
        w, l = self._get(winner), self._get(loser)
        self._apply(w, l.rating, l.rd, score=1.0)
        self._apply(l, w.rating, w.rd, score=0.0)
        w.games += 1
        l.games += 1

    def _apply(self, p: GlickoRating, opp_r: float, opp_rd: float, score: float) -> None:
        g = _g(opp_rd)
        e = _expected(p.rating, opp_r, opp_rd)
        d2_inv = _Q * _Q * g * g * e * (1 - e)
        if d2_inv <= 0:
            return
        d2 = 1.0 / d2_inv
        denom = 1.0 / (p.rd * p.rd) + 1.0 / d2
        new_rd = math.sqrt(1.0 / denom)
        p.rating += (_Q / denom) * g * (score - e)
        p.rd = max(30.0, new_rd)  # floor so a player never looks perfectly known

    def train(self, results) -> "GlickoModel":
        """Fit ratings from settled outcomes (odds not required)."""
        for res in results:
            winner_sel = next(
                (w for w in res.outcomes.values() if w in ("home", "away")), None
            )
            if winner_sel == "home":
                self.update(res.match.home, res.match.away)
            elif winner_sel == "away":
                self.update(res.match.away, res.match.home)
        return self

    def win_probability(self, player: str, opponent: str) -> float:
        p, o = self._get(player), self._get(opponent)
        # Combine both deviations so two uncertain players widen the estimate.
        combined_rd = math.sqrt(p.rd * p.rd + o.rd * o.rd)
        return _expected(p.rating, o.rating, combined_rd)

    def market_match_winner(self, home: str, away: str) -> ModelOutput:
        p, o = self._get(home), self._get(away)
        if p.games == 0 or o.games == 0:
            raise InsufficientModelData(
                f"unrated player(s): {home} games={p.games}, {away} games={o.games}"
            )
        ph = self.win_probability(home, away)
        # Map combined RD to a probability band: more uncertain ratings -> wider.
        combined_rd = math.sqrt(p.rd * p.rd + o.rd * o.rd)
        half = min(0.5, combined_rd / 1400.0)
        return ModelOutput(
            market="match_winner",
            probabilities={"home": ph, "away": 1.0 - ph},
            confidence_half_width=half,
            effective_samples=min(p.games, o.games),
            notes=[
                f"Glicko: {home} {p.rating:.0f}±{p.rd:.0f} vs "
                f"{away} {o.rating:.0f}±{o.rd:.0f}"
            ],
        )
