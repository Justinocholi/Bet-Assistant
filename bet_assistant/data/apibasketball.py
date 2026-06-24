"""API-Basketball data provider (https://www.api-basketball.com).

Same API-Sports family as the football adapter, so it follows the same rules:
stdlib injectable transport, ``ProviderError`` on failure, missing data flagged
rather than fabricated. It maps games + moneyline odds + team scoring form, and
backfills finished games for backtesting.

Basketball strength is best modelled with Elo/Glicko trained on results, so the
primary product here is ``get_results`` (settled games for training/backtest);
``get_fixtures`` provides upcoming games with the moneyline market and scoring
form for display and value assessment.
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
from .schema import DataQuality, Match, OddsLine, Sport, TeamForm

_BASE_URL = "https://v1.basketball.api-sports.io"
_MIN_FORM_GAMES = 5


class APIBasketballProvider(DataProvider):
    def __init__(
        self,
        api_key: str,
        league: int,
        season,  # api-basketball seasons are strings like "2023-2024"
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

    def _get(self, path: str, params: dict):
        url = build_url(self.base_url, path, params)
        headers = {"x-apisports-key": self.api_key, "Accept": "application/json"}
        try:
            payload = self._transport(url, headers)
        except HttpError as exc:
            raise ProviderError(str(exc)) from exc
        if not isinstance(payload, dict):
            raise ProviderError(f"unexpected payload from {path}")
        if payload.get("errors"):
            raise ProviderError(f"API-Basketball error on {path}: {payload['errors']}")
        response = payload.get("response")
        if response is None:
            raise ProviderError(f"missing 'response' from {path}")
        return response

    def get_fixtures(self, sport: Sport, on: date) -> list[Match]:
        if sport is not Sport.BASKETBALL:
            raise ProviderError(
                f"APIBasketballProvider supports basketball only; got {sport.value}."
            )
        raw = self._get(
            "games",
            {"date": on.isoformat(), "league": self.league, "season": self.season},
        )
        matches: list[Match] = []
        for g in raw or []:
            try:
                matches.append(self._build_match(g, on))
            except InsufficientDataError:
                continue
        return matches

    def _build_match(self, g: dict, on: date) -> Match:
        teams = g.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        home_id, home_name = home.get("id"), home.get("name")
        away_id, away_name = away.get("id"), away.get("name")
        game_id = g.get("id")
        if not (home_id and away_id and home_name and away_name and game_id):
            raise InsufficientDataError("game missing team/id fields")

        missing: list[str] = []
        home_form = self._team_form(home_id, home_name, missing, "home_form")
        away_form = self._team_form(away_id, away_name, missing, "away_form")
        effective = min(
            home_form.matches_played if home_form else 0,
            away_form.matches_played if away_form else 0,
        )
        quality = DataQuality(effective_samples=effective, missing_fields=missing)
        match = Match(
            sport=Sport.BASKETBALL,
            home=home_name,
            away=away_name,
            match_date=on,
            home_form=home_form,
            away_form=away_form,
            quality=quality,
        )
        line = self._moneyline_odds(game_id)
        if line is not None:
            match.odds.append(line)
        else:
            quality.with_missing("odds")
        return match

    def _team_form(self, team_id, team_name, missing, field) -> Optional[TeamForm]:
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

        games = (stats.get("games") or {}).get("played") or {}
        played = _as_int(games.get("all"))
        if played < _MIN_FORM_GAMES:
            missing.append(field)
            return None

        points = stats.get("points") or {}
        pf = (points.get("for") or {}).get("average") or {}
        pa = (points.get("against") or {}).get("average") or {}
        pf_home, pf_away = _as_float(pf.get("home")), _as_float(pf.get("away"))
        pa_home, pa_away = _as_float(pa.get("home")), _as_float(pa.get("away"))
        if pf_home == 0 and pf_away == 0 and pa_home == 0 and pa_away == 0:
            missing.append(field)
            return None

        return TeamForm(
            team=team_name,
            matches_played=played,
            goals_for_home=pf_home or pf_away,
            goals_against_home=pa_home or pa_away,
            goals_for_away=pf_away or pf_home,
            goals_against_away=pa_away or pa_home,
        )

    def _moneyline_odds(self, game_id) -> Optional[OddsLine]:
        try:
            resp = self._get("odds", {"game": game_id})
        except ProviderError:
            return None
        for entry in resp or []:
            for book in entry.get("bookmakers") or []:
                for bet in book.get("bets") or []:
                    selections: dict[str, float] = {}
                    for v in bet.get("values") or []:
                        label = str(v.get("value", "")).lower()
                        odd = _as_float(v.get("odd"))
                        if odd <= 1.0:
                            continue
                        if label in ("home", "1"):
                            selections["home"] = odd
                        elif label in ("away", "2"):
                            selections["away"] = odd
                    if {"home", "away"} <= set(selections):
                        return OddsLine(
                            market="moneyline",
                            selections={k: selections[k] for k in ("home", "away")},
                            bookmaker=str(book.get("name", "unknown")),
                        )
        return None

    def get_results(self, sport: Sport, start: date, end: date) -> list[MatchResult]:
        """Backfill finished games as two-way (home/away) settled results."""
        if sport is not Sport.BASKETBALL:
            raise ProviderError(
                f"APIBasketballProvider supports basketball only; got {sport.value}."
            )
        results: list[MatchResult] = []
        cur = start
        # api-basketball games are queried per-date; walk the range day by day.
        from datetime import timedelta

        while cur <= end:
            raw = self._get(
                "games",
                {"date": cur.isoformat(), "league": self.league, "season": self.season},
            )
            for g in raw or []:
                status = ((g.get("status") or {}).get("short") or "").upper()
                if status not in ("FT", "AOT"):  # finished / after overtime
                    continue
                teams = g.get("teams") or {}
                home = (teams.get("home") or {}).get("name")
                away = (teams.get("away") or {}).get("name")
                scores = g.get("scores") or {}
                hs = (scores.get("home") or {}).get("total")
                as_ = (scores.get("away") or {}).get("total")
                if home is None or away is None or hs is None or as_ is None:
                    continue
                winner = "home" if _as_int(hs) > _as_int(as_) else "away"
                match = Match(
                    sport=Sport.BASKETBALL,
                    home=home,
                    away=away,
                    match_date=cur,
                    quality=DataQuality(effective_samples=0),
                )
                results.append(MatchResult(match, {"moneyline": winner}))
            cur += timedelta(days=1)
        return results


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
