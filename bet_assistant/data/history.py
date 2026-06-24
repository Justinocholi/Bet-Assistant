"""Historical results: score -> market outcomes, and point-in-time form.

Two pieces a real backtest needs and that are easy to get wrong:

1. ``outcomes_from_score`` maps a final score to the winning selection of each
   market (1X2, Over/Under, BTTS), so settled fixtures can be graded.

2. ``build_pointintime_results`` reconstructs each fixture's team form from
   **only the matches that happened before it**. Using season-aggregate stats
   (which include the match being predicted) leaks the future and inflates
   backtest results — the classic mistake. This builder avoids that, so a
   backtest reflects what the model could actually have known at kickoff.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
from typing import Deque, Optional

from .providers import MatchResult
from .schema import DataQuality, Match, OddsLine, Sport, TeamForm


@dataclass
class ScoredFixture:
    """A finished match with its final score and (optionally) closing odds."""

    match_date: date
    home: str
    away: str
    home_score: int
    away_score: int
    # market -> {selection: decimal_odds}; the closing line, if captured.
    odds: dict[str, dict[str, float]] = field(default_factory=dict)


def outcomes_from_score(
    home_score: int, away_score: int, ou_line: float = 2.5
) -> dict[str, str]:
    """Grade the standard football markets from a final score."""
    total = home_score + away_score
    if home_score > away_score:
        winner = "home"
    elif home_score < away_score:
        winner = "away"
    else:
        winner = "draw"
    return {
        "1x2": winner,
        f"over_under_{ou_line}": "over" if total > ou_line else "under",
        "btts": "yes" if (home_score > 0 and away_score > 0) else "no",
    }


@dataclass
class _RollingTeam:
    """Per-venue rolling tallies for one team, updated as fixtures are walked."""

    home_for: Deque[float] = field(default_factory=lambda: deque(maxlen=None))
    home_against: Deque[float] = field(default_factory=lambda: deque(maxlen=None))
    away_for: Deque[float] = field(default_factory=lambda: deque(maxlen=None))
    away_against: Deque[float] = field(default_factory=lambda: deque(maxlen=None))
    recent: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    played: int = 0
    last_date: Optional[date] = None

    def configure(self, window: Optional[int]) -> None:
        for dq_name in ("home_for", "home_against", "away_for", "away_against"):
            dq: Deque = getattr(self, dq_name)
            if dq.maxlen != window:
                setattr(self, dq_name, deque(dq, maxlen=window))


def _avg(values, fallback: float) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else fallback


def build_pointintime_results(
    fixtures: list[ScoredFixture],
    sport: Sport = Sport.FOOTBALL,
    window: Optional[int] = None,
    min_prior: int = 5,
    league_avg_goals: float = 1.35,
) -> list[MatchResult]:
    """Walk fixtures in date order, attaching form built only from the past.

    ``window`` bounds the rolling window (None = season-to-date). ``min_prior``
    is how many prior matches a team needs before we trust its rates; below
    that the form is still provided but ``effective_samples`` stays low, so the
    value gate will refuse to bet.
    """
    ordered = sorted(fixtures, key=lambda f: (f.match_date, f.home, f.away))
    teams: dict[str, _RollingTeam] = defaultdict(_RollingTeam)
    results: list[MatchResult] = []

    for fx in ordered:
        h = teams[fx.home]
        a = teams[fx.away]
        h.configure(window)
        a.configure(window)

        home_form = _snapshot(fx.home, h, league_avg_goals)
        away_form = _snapshot(fx.away, a, league_avg_goals)
        effective = min(h.played, a.played)

        match = Match(
            sport=sport,
            home=fx.home,
            away=fx.away,
            match_date=fx.match_date,
            home_form=home_form,
            away_form=away_form,
            quality=DataQuality(effective_samples=effective),
        )
        # Rest days from each side's previous fixture (point-in-time).
        if home_form is not None and h.last_date is not None:
            home_form.rest_days = (fx.match_date - h.last_date).days
        if away_form is not None and a.last_date is not None:
            away_form.rest_days = (fx.match_date - a.last_date).days

        for market, selections in fx.odds.items():
            match.odds.append(
                OddsLine(market=market, selections=selections,
                         bookmaker="historical", is_closing_line=True)
            )

        results.append(MatchResult(match, outcomes_from_score(
            fx.home_score, fx.away_score)))

        # --- now update rolling state WITH this fixture (for future matches) ---
        h.home_for.append(fx.home_score)
        h.home_against.append(fx.away_score)
        a.away_for.append(fx.away_score)
        a.away_against.append(fx.home_score)
        _push_result(h, fx.home_score, fx.away_score)
        _push_result(a, fx.away_score, fx.home_score)
        h.played += 1
        a.played += 1
        h.last_date = fx.match_date
        a.last_date = fx.match_date

    return results


def _snapshot(name: str, t: _RollingTeam, league_avg: float) -> Optional[TeamForm]:
    if t.played == 0:
        return None  # no prior data at all -> model declares insufficient data
    return TeamForm(
        team=name,
        matches_played=t.played,
        goals_for_home=_avg(t.home_for, league_avg),
        goals_against_home=_avg(t.home_against, league_avg),
        goals_for_away=_avg(t.away_for, league_avg),
        goals_against_away=_avg(t.away_against, league_avg),
        recent_results=list(t.recent),
    )


def _push_result(t: _RollingTeam, scored: int, conceded: int) -> None:
    t.recent.append("W" if scored > conceded else "D" if scored == conceded else "L")
