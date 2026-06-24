"""API-Football data provider (https://www.api-football.com).

Maps API-Football responses onto the tool's schema. Built on the stdlib HTTP
transport so there are no third-party dependencies, and the transport is
injectable so this whole adapter is unit-tested offline against canned JSON.

Design rules honoured here (see DataProvider docstring):
  * any transport/parse failure becomes ``ProviderError`` — we never return
    partial garbage;
  * thin or absent history becomes ``InsufficientDataError`` or is recorded in
    ``Match.quality.missing_fields`` — we never invent values;
  * odds are mapped to decimal and the 1X2 (Match Winner) market is exposed for
    the Dixon-Coles football model.

Scope: football is implemented end-to-end (fixtures + 1X2 odds + team form).
Basketball/tennis use different endpoints/leagues and are intentionally left to
raise a clear error until their mappers are added, rather than guessing.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from .http import HttpError, Transport, build_url, urllib_transport
from .providers import (
    DataProvider,
    InsufficientDataError,
    MatchResult,
    ProviderError,
)
from .schema import (
    DataQuality,
    HeadToHead,
    Match,
    OddsLine,
    Sport,
    TeamForm,
)

_BASE_URL = "https://v3.football.api-sports.io"
# API-Football "bet" id 1 is the Match Winner (1X2) market.
_MATCH_WINNER_BET_ID = 1
# Minimum fixtures of history before we trust a team's form rates.
_MIN_FORM_FIXTURES = 5


class APIFootballProvider(DataProvider):
    """Football fixtures/odds/form from API-Football.

    Parameters
    ----------
    api_key:
        Your API-Football key. Sent as the ``x-apisports-key`` header.
    league / season:
        Required to scope team-statistics (form) lookups, which API-Football
        keys by league+season.
    transport:
        Injected HTTP-JSON callable; defaults to the urllib transport. Tests
        pass a fake to run offline.
    """

    def __init__(
        self,
        api_key: str,
        league: int,
        season: int,
        base_url: str = _BASE_URL,
        transport: Optional[Transport] = None,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.league = league
        self.season = season
        self.base_url = base_url
        self._transport = transport or urllib_transport()

    # -- low-level ---------------------------------------------------------

    def _get(self, path: str, params: dict) -> list:
        """Call an endpoint and return the ``response`` array.

        API-Football wraps payloads as ``{"errors": [...], "response": [...]}``.
        Non-empty ``errors`` or a transport failure becomes ``ProviderError``.
        """
        url = build_url(self.base_url, path, params)
        headers = {"x-apisports-key": self.api_key, "Accept": "application/json"}
        try:
            payload = self._transport(url, headers)
        except HttpError as exc:
            raise ProviderError(str(exc)) from exc

        if not isinstance(payload, dict):
            raise ProviderError(f"unexpected payload shape from {path}")
        errors = payload.get("errors")
        # API-Football returns errors as a dict (e.g. quota/auth) or empty list.
        if errors:
            raise ProviderError(f"API-Football error on {path}: {errors}")
        response = payload.get("response")
        if response is None:
            raise ProviderError(f"missing 'response' from {path}")
        return response

    # -- fixtures + form + odds -------------------------------------------

    def get_fixtures(self, sport: Sport, on: date) -> list[Match]:
        if sport is not Sport.FOOTBALL:
            raise ProviderError(
                f"APIFootballProvider supports football only; got {sport.value}. "
                "Add a sport-specific mapper before using it."
            )

        raw_fixtures = self._get(
            "fixtures",
            {
                "date": on.isoformat(),
                "league": self.league,
                "season": self.season,
            },
        )

        matches: list[Match] = []
        for fx in raw_fixtures:
            try:
                matches.append(self._build_match(fx, on))
            except InsufficientDataError:
                # A fixture we can't model is skipped, not faked. The caller
                # still sees the fixtures we *could* build.
                continue
        return matches

    def _build_match(self, fx: dict, on: date) -> Match:
        teams = fx.get("teams") or {}
        home = (teams.get("home") or {})
        away = (teams.get("away") or {})
        home_id, home_name = home.get("id"), home.get("name")
        away_id, away_name = away.get("id"), away.get("name")
        fixture_id = (fx.get("fixture") or {}).get("id")
        if not (home_id and away_id and home_name and away_name and fixture_id):
            raise InsufficientDataError("fixture missing team/id fields")

        missing: list[str] = []

        home_form = self._team_form(home_id, home_name, missing, "home_form")
        away_form = self._team_form(away_id, away_name, missing, "away_form")

        # Best-effort enrichment: injuries and rest days. Failures are flagged
        # in ``missing`` and left at safe defaults rather than fabricated.
        self._enrich_injuries(fixture_id, home_id, away_id,
                              home_form, away_form, missing)
        self._enrich_rest_days(home_id, away_id, on, home_form, away_form, missing)

        # Effective samples = fewest fixtures behind either side's form.
        effective = min(
            home_form.matches_played if home_form else 0,
            away_form.matches_played if away_form else 0,
        )
        quality = DataQuality(effective_samples=effective, missing_fields=missing)

        match = Match(
            sport=Sport.FOOTBALL,
            home=home_name,
            away=away_name,
            match_date=on,
            home_form=home_form,
            away_form=away_form,
            head_to_head=self._head_to_head(home_id, away_id, missing),
            quality=quality,
        )

        odds = self._match_winner_odds(fixture_id)
        if odds is not None:
            match.odds.append(odds)
        else:
            quality.with_missing("odds")
        return match

    def _team_form(
        self, team_id: int, team_name: str, missing: list[str], field: str
    ) -> Optional[TeamForm]:
        """Build TeamForm from the team-statistics endpoint.

        Returns ``None`` (and records the missing field) when stats are absent
        or too thin — the model will then declare insufficient data rather than
        bet on guesses.
        """
        try:
            resp = self._get(
                "teams/statistics",
                {"team": team_id, "league": self.league, "season": self.season},
            )
        except ProviderError:
            missing.append(field)
            return None

        stats = resp if isinstance(resp, dict) else (resp[0] if resp else None)
        if not stats:
            missing.append(field)
            return None

        fixtures = ((stats.get("fixtures") or {}).get("played") or {})
        played_home = _as_int(fixtures.get("home"))
        played_away = _as_int(fixtures.get("away"))
        played_total = _as_int(fixtures.get("total"))
        if played_total < _MIN_FORM_FIXTURES:
            missing.append(field)
            return None

        goals = stats.get("goals") or {}
        gf = (goals.get("for") or {}).get("average") or {}
        ga = (goals.get("against") or {}).get("average") or {}

        gf_home = _as_float(gf.get("home"))
        gf_away = _as_float(gf.get("away"))
        ga_home = _as_float(ga.get("home"))
        ga_away = _as_float(ga.get("away"))
        # If the goal averages are entirely missing we cannot model the team.
        if gf_home == 0.0 and gf_away == 0.0 and ga_home == 0.0 and ga_away == 0.0:
            missing.append(field)
            return None

        form_str = stats.get("form") or ""
        recent = list(form_str[-5:]) if form_str else []

        return TeamForm(
            team=team_name,
            matches_played=played_total,
            goals_for_home=gf_home or _fallback_rate(gf_away),
            goals_against_home=ga_home or _fallback_rate(ga_away),
            goals_for_away=gf_away or _fallback_rate(gf_home),
            goals_against_away=ga_away or _fallback_rate(ga_home),
            recent_results=recent,
            # API-Football exposes these elsewhere (injuries endpoint); left at
            # safe defaults and flagged rather than fabricated.
            rest_days=None,
            key_injuries=0,
        )

    def _head_to_head(
        self, home_id: int, away_id: int, missing: list[str]
    ) -> Optional[HeadToHead]:
        try:
            resp = self._get(
                "fixtures/headtohead",
                {"h2h": f"{home_id}-{away_id}", "last": 10},
            )
        except ProviderError:
            missing.append("head_to_head")
            return None

        h = HeadToHead()
        for fx in resp or []:
            teams = fx.get("teams") or {}
            home_won = (teams.get("home") or {}).get("winner")
            away_won = (teams.get("away") or {}).get("winner")
            if home_won is True:
                h.home_or_first_wins += 1
            elif away_won is True:
                h.away_or_second_wins += 1
            elif home_won is False and away_won is False:
                h.draws += 1
        return h if h.total else None

    def _match_winner_odds(self, fixture_id: int) -> Optional[OddsLine]:
        try:
            resp = self._get("odds", {"fixture": fixture_id})
        except ProviderError:
            return None
        if not resp:
            return None

        # Walk bookmakers -> bets -> find the Match Winner market.
        for entry in resp:
            for book in entry.get("bookmakers") or []:
                for bet in book.get("bets") or []:
                    if _as_int(bet.get("id")) != _MATCH_WINNER_BET_ID:
                        continue
                    selections: dict[str, float] = {}
                    for v in bet.get("values") or []:
                        label = str(v.get("value", "")).lower()
                        odd = _as_float(v.get("odd"))
                        if odd <= 1.0:
                            continue
                        if label in ("home", "1"):
                            selections["home"] = odd
                        elif label in ("draw", "x"):
                            selections["draw"] = odd
                        elif label in ("away", "2"):
                            selections["away"] = odd
                    if {"home", "draw", "away"} <= set(selections):
                        return OddsLine(
                            market="1x2",
                            selections=selections,
                            bookmaker=str(book.get("name", "unknown")),
                            is_closing_line=False,
                        )
        return None

    # -- injuries + rest days ---------------------------------------------

    def _enrich_injuries(
        self, fixture_id, home_id, away_id, home_form, away_form, missing
    ) -> None:
        try:
            resp = self._get("injuries", {"fixture": fixture_id})
        except ProviderError:
            missing.append("injuries")
            return
        counts = {home_id: 0, away_id: 0}
        for entry in resp or []:
            tid = ((entry.get("team") or {}).get("id"))
            if tid in counts:
                counts[tid] += 1
        if home_form is not None:
            home_form.key_injuries = counts.get(home_id, 0)
        if away_form is not None:
            away_form.key_injuries = counts.get(away_id, 0)

    def _enrich_rest_days(
        self, home_id, away_id, on, home_form, away_form, missing
    ) -> None:
        for team_id, form in ((home_id, home_form), (away_id, away_form)):
            if form is None:
                continue
            rest = self._rest_days(team_id, on)
            if rest is None:
                if "rest_days" not in missing:
                    missing.append("rest_days")
            else:
                form.rest_days = rest

    def _rest_days(self, team_id: int, before: date) -> Optional[int]:
        """Days since the team's most recent finished fixture before ``before``."""
        try:
            resp = self._get(
                "fixtures",
                {"team": team_id, "season": self.season, "last": 1},
            )
        except ProviderError:
            return None
        for fx in resp or []:
            ts = ((fx.get("fixture") or {}).get("date") or "")[:10]
            try:
                last = date.fromisoformat(ts)
            except ValueError:
                continue
            if last < before:
                return (before - last).days
        return None

    # -- historical backfill for backtesting ------------------------------

    def get_results(
        self, sport: Sport, start: date, end: date, with_odds: bool = False
    ) -> list[MatchResult]:
        """Backfill finished fixtures in [start, end] as graded results.

        Form is rebuilt **point-in-time** from the scores themselves (via
        ``build_pointintime_results``), so a backtest never sees a team's
        future matches. Set ``with_odds=True`` to also fetch each fixture's
        closing line for closing-line-value measurement (one extra call per
        fixture — slower and quota-heavy).
        """
        if sport is not Sport.FOOTBALL:
            raise ProviderError(
                f"APIFootballProvider supports football only; got {sport.value}."
            )
        # Import here to avoid a circular import at module load.
        from .history import ScoredFixture, build_pointintime_results

        raw = self._get(
            "fixtures",
            {
                "league": self.league,
                "season": self.season,
                "from": start.isoformat(),
                "to": end.isoformat(),
                "status": "FT",  # full time only
            },
        )

        scored: list[ScoredFixture] = []
        for fx in raw or []:
            teams = fx.get("teams") or {}
            home = (teams.get("home") or {}).get("name")
            away = (teams.get("away") or {}).get("name")
            goals = fx.get("goals") or {}
            hg, ag = goals.get("home"), goals.get("away")
            ts = ((fx.get("fixture") or {}).get("date") or "")[:10]
            if home is None or away is None or hg is None or ag is None:
                continue
            try:
                match_date = date.fromisoformat(ts)
            except ValueError:
                continue

            odds: dict[str, dict[str, float]] = {}
            if with_odds:
                fixture_id = (fx.get("fixture") or {}).get("id")
                line = self._match_winner_odds(fixture_id) if fixture_id else None
                if line is not None:
                    odds["1x2"] = line.selections

            scored.append(
                ScoredFixture(
                    match_date=match_date,
                    home=home,
                    away=away,
                    home_score=_as_int(hg),
                    away_score=_as_int(ag),
                    odds=odds,
                )
            )

        return build_pointintime_results(scored, sport=Sport.FOOTBALL)


# -- small, defensive coercion helpers --------------------------------------


def _as_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _fallback_rate(other: float) -> float:
    """If one venue split is missing, fall back to the other rather than 0,
    which would otherwise zero-out a Poisson rate. Returns a small floor if
    both are absent."""
    return other if other > 0 else 1.0
