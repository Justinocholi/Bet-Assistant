"""Typed data structures shared across the tool.

These intentionally model *uncertainty about the data itself*: every record
carries a notion of how much it can be trusted, so downstream models can refuse
to bet when the inputs are thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class Sport(str, Enum):
    FOOTBALL = "football"
    BASKETBALL = "basketball"
    TENNIS = "tennis"


@dataclass
class DataQuality:
    """How trustworthy a bundle of inputs is.

    ``effective_samples`` is the number of relevant observations behind the
    estimate (e.g. matches in form window with valid data). ``missing_fields``
    lists inputs that could not be loaded — their presence forces extra caution.
    """

    effective_samples: int
    missing_fields: list[str] = field(default_factory=list)

    @property
    def is_sufficient_for(self) -> bool:
        # A coarse gate; the real threshold lives in ValueConfig and is applied
        # by the pipeline. Kept here for quick checks.
        return self.effective_samples > 0

    def with_missing(self, name: str) -> "DataQuality":
        if name not in self.missing_fields:
            self.missing_fields.append(name)
        return self


@dataclass
class TeamForm:
    """Recent form and splits for a team (football / basketball)."""

    team: str
    matches_played: int
    # Per-game scoring rates, separated by venue where available.
    goals_for_home: float
    goals_against_home: float
    goals_for_away: float
    goals_against_away: float
    recent_results: list[str] = field(default_factory=list)  # e.g. ["W","D","L"]
    rest_days: Optional[int] = None
    key_injuries: int = 0  # count of significant absences


@dataclass
class PlayerForm:
    """Recent form for a tennis player."""

    player: str
    matches_played: int
    win_rate_surface: float  # win rate on the relevant surface
    recent_results: list[str] = field(default_factory=list)
    rest_days: Optional[int] = None
    injury_flag: bool = False


@dataclass
class HeadToHead:
    home_or_first_wins: int = 0
    draws: int = 0
    away_or_second_wins: int = 0

    @property
    def total(self) -> int:
        return self.home_or_first_wins + self.draws + self.away_or_second_wins


@dataclass
class OddsLine:
    """A bookmaker's quoted decimal odds for the selections of one market.

    ``selections`` maps selection name -> decimal odds (e.g.
    {"home": 2.10, "draw": 3.40, "away": 3.60}).
    """

    market: str
    selections: dict[str, float]
    bookmaker: str = "unknown"
    is_closing_line: bool = False


@dataclass
class Match:
    """A fixture plus everything known about it."""

    sport: Sport
    home: str  # home team or first player
    away: str  # away team or second player
    match_date: date
    home_form: Optional[TeamForm] = None
    away_form: Optional[TeamForm] = None
    home_player: Optional[PlayerForm] = None
    away_player: Optional[PlayerForm] = None
    head_to_head: Optional[HeadToHead] = None
    odds: list[OddsLine] = field(default_factory=list)
    quality: DataQuality = field(default_factory=lambda: DataQuality(0))

    def odds_for(self, market: str) -> Optional[OddsLine]:
        for line in self.odds:
            if line.market == market:
                return line
        return None
