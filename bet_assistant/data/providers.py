"""Data providers.

The real world uses an HTTP sports-data API (API-Football, Sportradar, ...).
That belongs behind the ``DataProvider`` interface so the rest of the tool never
depends on a specific vendor and can be tested offline.

Two failure modes are first-class:

* ``ProviderError`` — the upstream API failed (timeout, 5xx, auth). Callers
  should degrade gracefully, never crash, and never fabricate data.
* ``InsufficientDataError`` — the API responded but there isn't enough relevant
  history to model the match. This is a *normal* outcome ("no bet"), not a bug.

``MockProvider`` generates deterministic synthetic fixtures so the whole
pipeline, backtest and CLI demo run with no network and no API key.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Iterable, Optional

from .schema import (
    DataQuality,
    HeadToHead,
    Match,
    OddsLine,
    PlayerForm,
    Sport,
    TeamForm,
)


class ProviderError(RuntimeError):
    """Upstream data API failed. Degrade gracefully; do not fabricate."""


class InsufficientDataError(RuntimeError):
    """Not enough relevant data to model this match — a valid 'no bet' state."""


class DataProvider:
    """Interface every concrete provider implements.

    Implementations must:
      * raise ``ProviderError`` on transport/auth/parse failure (never return
        partial garbage),
      * populate ``Match.quality`` honestly, including ``missing_fields``,
      * never invent values for missing inputs.
    """

    def get_fixtures(self, sport: Sport, on: date) -> list[Match]:  # pragma: no cover
        raise NotImplementedError

    def get_results(
        self, sport: Sport, start: date, end: date
    ) -> list["MatchResult"]:  # pragma: no cover
        raise NotImplementedError


class MatchResult:
    """A settled historical match used for backtesting/calibration."""

    def __init__(self, match: Match, outcomes: dict[str, str]):
        # ``outcomes`` maps market -> winning selection, e.g.
        # {"1x2": "home", "over_under_2.5": "over", "btts": "yes"}.
        self.match = match
        self.outcomes = outcomes


def safe_get_fixtures(
    provider: DataProvider, sport: Sport, on: date
) -> tuple[list[Match], Optional[str]]:
    """Wrapper that turns provider failure into a degraded, non-crashing result.

    Returns ``(matches, error_message)``. On failure, ``matches`` is empty and
    ``error_message`` explains why — the caller surfaces it instead of guessing.
    """
    try:
        return provider.get_fixtures(sport, on), None
    except ProviderError as exc:
        return [], f"Data provider unavailable for {sport.value} on {on}: {exc}"
    except Exception as exc:  # defensive: never let a provider crash the tool
        return [], f"Unexpected provider failure for {sport.value} on {on}: {exc!r}"


# ---------------------------------------------------------------------------
# Mock provider — deterministic synthetic data for offline runs and tests.
# ---------------------------------------------------------------------------


class MockProvider(DataProvider):
    """Generates plausible, deterministic fixtures with realistic vig in odds.

    The ``fail_rate`` and ``thin_rate`` knobs let tests exercise the graceful
    degradation paths (API failure, insufficient data).
    """

    def __init__(self, seed: int = 7, fail_rate: float = 0.0, thin_rate: float = 0.0):
        self._seed = seed
        self.fail_rate = fail_rate
        self.thin_rate = thin_rate

    def _rng(self, salt: str) -> random.Random:
        return random.Random(f"{self._seed}:{salt}")

    def get_fixtures(self, sport: Sport, on: date) -> list[Match]:
        rng = self._rng(f"fixtures:{sport.value}:{on.isoformat()}")
        if rng.random() < self.fail_rate:
            raise ProviderError("simulated upstream API failure")

        n = rng.randint(3, 6)
        matches = []
        for i in range(n):
            matches.append(self._make_match(sport, on, rng, i))
        return matches

    def _make_match(
        self, sport: Sport, on: date, rng: random.Random, i: int
    ) -> Match:
        thin = rng.random() < self.thin_rate
        home, away = f"{sport.value.title()}_A{i}", f"{sport.value.title()}_B{i}"

        if sport is Sport.TENNIS:
            samples = 5 if thin else rng.randint(25, 60)
            quality = DataQuality(effective_samples=samples)
            hp = PlayerForm(
                player=home,
                matches_played=samples,
                win_rate_surface=rng.uniform(0.45, 0.70),
                recent_results=rng.choices(["W", "L"], k=5),
                rest_days=rng.randint(1, 7),
            )
            ap = PlayerForm(
                player=away,
                matches_played=samples,
                win_rate_surface=rng.uniform(0.40, 0.65),
                recent_results=rng.choices(["W", "L"], k=5),
                rest_days=rng.randint(1, 7),
            )
            match = Match(
                sport=sport,
                home=home,
                away=away,
                match_date=on,
                home_player=hp,
                away_player=ap,
                head_to_head=HeadToHead(rng.randint(0, 5), 0, rng.randint(0, 5)),
                quality=quality,
            )
            self._add_two_way_odds(match, rng)
            return match

        # Football / basketball: team form
        samples = 4 if thin else rng.randint(20, 38)
        quality = DataQuality(effective_samples=samples)
        scale = 1.4 if sport is Sport.FOOTBALL else 105.0  # goals vs points
        spread = 0.4 if sport is Sport.FOOTBALL else 8.0
        hf = TeamForm(
            team=home,
            matches_played=samples,
            goals_for_home=rng.uniform(scale, scale + spread),
            goals_against_home=rng.uniform(scale - spread, scale),
            goals_for_away=rng.uniform(scale - spread, scale),
            goals_against_away=rng.uniform(scale, scale + spread),
            recent_results=rng.choices(["W", "D", "L"], k=5),
            rest_days=rng.randint(2, 7),
            key_injuries=rng.randint(0, 2),
        )
        af = TeamForm(
            team=away,
            matches_played=samples,
            goals_for_home=rng.uniform(scale, scale + spread),
            goals_against_home=rng.uniform(scale - spread, scale),
            goals_for_away=rng.uniform(scale - spread, scale),
            goals_against_away=rng.uniform(scale, scale + spread),
            recent_results=rng.choices(["W", "D", "L"], k=5),
            rest_days=rng.randint(2, 7),
            key_injuries=rng.randint(0, 2),
        )
        match = Match(
            sport=sport,
            home=home,
            away=away,
            match_date=on,
            home_form=hf,
            away_form=af,
            head_to_head=HeadToHead(
                rng.randint(0, 4), rng.randint(0, 3), rng.randint(0, 4)
            ),
            quality=quality,
        )
        if sport is Sport.FOOTBALL:
            self._add_1x2_odds(match, rng)
        else:
            self._add_two_way_odds(match, rng)
        return match

    @staticmethod
    def _apply_vig(true_probs: dict[str, float], overround: float) -> dict[str, float]:
        """Turn fair probabilities into decimal odds carrying an overround."""
        return {k: round(1.0 / (p * overround), 2) for k, p in true_probs.items()}

    def _add_1x2_odds(self, match: Match, rng: random.Random) -> None:
        ph = rng.uniform(0.35, 0.55)
        pd = rng.uniform(0.20, 0.30)
        pa = max(0.05, 1.0 - ph - pd)
        total = ph + pd + pa
        fair = {"home": ph / total, "draw": pd / total, "away": pa / total}
        overround = rng.uniform(1.05, 1.08)  # ~5-8% margin
        match.odds.append(
            OddsLine(
                market="1x2",
                selections=self._apply_vig(fair, overround),
                bookmaker="mock",
                is_closing_line=True,
            )
        )

    def _add_two_way_odds(self, match: Match, rng: random.Random) -> None:
        ph = rng.uniform(0.40, 0.60)
        fair = {"home": ph, "away": 1.0 - ph}
        overround = rng.uniform(1.04, 1.07)
        match.odds.append(
            OddsLine(
                market="moneyline",
                selections=self._apply_vig(fair, overround),
                bookmaker="mock",
                is_closing_line=True,
            )
        )

    def get_results(
        self, sport: Sport, start: date, end: date
    ) -> list[MatchResult]:
        """Synthesise settled matches by sampling outcomes from the fair probs."""
        results: list[MatchResult] = []
        day = start
        while day <= end:
            for match in self.get_fixtures(sport, day):
                results.append(self._settle(match))
            day += timedelta(days=1)
        return results

    def _settle(self, match: Match) -> MatchResult:
        rng = self._rng(f"settle:{match.home}:{match.away}:{match.match_date}")
        line = match.odds[0]
        # Recover fair probabilities by stripping the overround, then sample.
        implied = {k: 1.0 / o for k, o in line.selections.items()}
        z = sum(implied.values())
        fair = {k: v / z for k, v in implied.items()}
        roll = rng.random()
        cum = 0.0
        winner = list(fair)[-1]
        for sel, p in fair.items():
            cum += p
            if roll <= cum:
                winner = sel
                break
        return MatchResult(match, {line.market: winner})
