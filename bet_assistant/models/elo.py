"""Elo ratings for relative strength (basketball, football moneyline).

Elo is the workhorse baseline for head-to-head relative strength. Each team has
a rating; the expected win probability is a logistic function of the rating
difference (plus home advantage). After each result, ratings move toward the
outcome by ``k`` times the surprise.

We expose both the rating store (trained from history) and a per-match
probability with an uncertainty band that shrinks as a team accrues games.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import InsufficientModelData, ModelOutput
from .mathutils import wilson_half_width


@dataclass
class EloRating:
    rating: float = 1500.0
    games: int = 0


@dataclass
class EloModel:
    k: float = 20.0
    home_advantage: float = 65.0  # rating points added to the home side
    scale: float = 400.0
    ratings: dict[str, EloRating] = field(default_factory=dict)

    def _get(self, team: str) -> EloRating:
        return self.ratings.setdefault(team, EloRating())

    def expected_home_win(self, home: str, away: str) -> float:
        rh = self._get(home).rating + self.home_advantage
        ra = self._get(away).rating
        return 1.0 / (1.0 + 10 ** (-(rh - ra) / self.scale))

    def update(self, home: str, away: str, home_won: bool) -> None:
        """Update ratings after a settled (two-way) result."""
        exp_home = self.expected_home_win(home, away)
        score_home = 1.0 if home_won else 0.0
        rh, ra = self._get(home), self._get(away)
        delta = self.k * (score_home - exp_home)
        rh.rating += delta
        ra.rating -= delta
        rh.games += 1
        ra.games += 1

    def train(self, results) -> "EloModel":
        """Fit ratings from an iterable of MatchResult (two-way moneyline)."""
        for res in results:
            line = res.match.odds[0]
            market = line.market
            winner = res.outcomes.get(market)
            if winner not in ("home", "away"):
                continue
            self.update(res.match.home, res.match.away, home_won=(winner == "home"))
        return self

    def market_moneyline(self, home: str, away: str) -> ModelOutput:
        rh, ra = self._get(home), self._get(away)
        if rh.games == 0 or ra.games == 0:
            raise InsufficientModelData(
                f"unrated team(s): {home} games={rh.games}, {away} games={ra.games}"
            )
        p_home = self.expected_home_win(home, away)
        n = min(rh.games, ra.games)
        return ModelOutput(
            market="moneyline",
            probabilities={"home": p_home, "away": 1.0 - p_home},
            confidence_half_width=wilson_half_width(max(p_home, 1 - p_home), n),
            effective_samples=n,
            notes=[
                f"Elo: {home} {rh.rating:.0f} vs {away} {ra.rating:.0f} "
                f"(+{self.home_advantage:.0f} home)"
            ],
        )
