"""Tennis data provider.

Unlike football/basketball, tennis is not covered by a single dominant
first-party API in the API-Sports family, and vendor schemas differ. So this
adapter is deliberately **configurable**: you pass the ``base_url`` and an
``auth_header`` for your chosen vendor, and the adapter maps a small, documented
JSON shape onto the tool's schema. The HTTP transport is injectable, so it is
fully unit-tested offline.

Expected response shapes (adapt the ``base_url``/paths to your vendor):

* ``GET {base}/games?date=YYYY-MM-DD`` -> ``{"response": [ {
      "id": <int>,
      "players": {"home": {"id": .., "name": ..},
                  "away": {"id": .., "name": ..}},
      "surface": "Hard" }, ... ]}``
* ``GET {base}/players/statistics?player=<id>&season=<s>&surface=<surface>``
   -> ``{"response": {"matches_played": <int>,
                       "surface_win_rate": <0..1>,
                       "form": "WLWW...", "injured": <bool>}}``
* ``GET {base}/odds?game=<id>`` -> bookmaker/bet/values with Home/Away prices.

Anything missing is flagged in ``Match.quality.missing_fields`` and never
invented; thin history lowers ``effective_samples`` so the value gate abstains.
The Glicko tennis model is best trained on results via ``get_results``.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from .http import HttpError, Transport, build_url, urllib_transport
from .providers import (
    DataProvider,
    InsufficientDataError,
    MatchResult,
    ProviderError,
)
from .schema import DataQuality, Match, OddsLine, PlayerForm, Sport

_MIN_FORM_MATCHES = 5


class APITennisProvider(DataProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        season,
        auth_header: str = "x-apisports-key",
        transport: Optional[Transport] = None,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        if not base_url:
            raise ValueError("base_url is required (tennis vendor varies)")
        self.api_key = api_key
        self.base_url = base_url
        self.season = season
        self.auth_header = auth_header
        self._transport = transport or urllib_transport()

    def _get(self, path: str, params: dict):
        url = build_url(self.base_url, path, params)
        headers = {self.auth_header: self.api_key, "Accept": "application/json"}
        try:
            payload = self._transport(url, headers)
        except HttpError as exc:
            raise ProviderError(str(exc)) from exc
        if not isinstance(payload, dict):
            raise ProviderError(f"unexpected payload from {path}")
        if payload.get("errors"):
            raise ProviderError(f"Tennis API error on {path}: {payload['errors']}")
        response = payload.get("response")
        if response is None:
            raise ProviderError(f"missing 'response' from {path}")
        return response

    def get_fixtures(self, sport: Sport, on: date) -> list[Match]:
        if sport is not Sport.TENNIS:
            raise ProviderError(
                f"APITennisProvider supports tennis only; got {sport.value}."
            )
        raw = self._get("games", {"date": on.isoformat()})
        matches: list[Match] = []
        for g in raw or []:
            try:
                matches.append(self._build_match(g, on))
            except InsufficientDataError:
                continue
        return matches

    def _build_match(self, g: dict, on: date) -> Match:
        players = g.get("players") or {}
        home = players.get("home") or {}
        away = players.get("away") or {}
        home_id, home_name = home.get("id"), home.get("name")
        away_id, away_name = away.get("id"), away.get("name")
        game_id = g.get("id")
        surface = g.get("surface")
        if not (home_id and away_id and home_name and away_name and game_id):
            raise InsufficientDataError("game missing player/id fields")

        missing: list[str] = []
        home_form = self._player_form(home_id, home_name, surface, missing, "home_player")
        away_form = self._player_form(away_id, away_name, surface, missing, "away_player")
        effective = min(
            home_form.matches_played if home_form else 0,
            away_form.matches_played if away_form else 0,
        )
        quality = DataQuality(effective_samples=effective, missing_fields=missing)

        match = Match(
            sport=Sport.TENNIS,
            home=home_name,
            away=away_name,
            match_date=on,
            home_player=home_form,
            away_player=away_form,
            quality=quality,
        )
        line = self._match_winner_odds(game_id)
        if line is not None:
            match.odds.append(line)
        else:
            quality.with_missing("odds")
        return match

    def _player_form(
        self, player_id, name, surface, missing, field
    ) -> Optional[PlayerForm]:
        try:
            resp = self._get(
                "players/statistics",
                {"player": player_id, "season": self.season, "surface": surface},
            )
        except ProviderError:
            missing.append(field)
            return None
        stats = resp if isinstance(resp, dict) else (resp[0] if resp else None)
        if not stats:
            missing.append(field)
            return None

        played = _as_int(stats.get("matches_played"))
        if played < _MIN_FORM_MATCHES:
            missing.append(field)
            return None
        win_rate = stats.get("surface_win_rate")
        if win_rate is None:
            missing.append(field)
            return None

        form_str = stats.get("form") or ""
        return PlayerForm(
            player=name,
            matches_played=played,
            win_rate_surface=_as_float(win_rate),
            recent_results=list(form_str[-5:]) if form_str else [],
            injury_flag=bool(stats.get("injured", False)),
        )

    def _match_winner_odds(self, game_id) -> Optional[OddsLine]:
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
                        if label in ("home", "1", "player 1"):
                            selections["home"] = odd
                        elif label in ("away", "2", "player 2"):
                            selections["away"] = odd
                    if {"home", "away"} <= set(selections):
                        return OddsLine(
                            market="match_winner",
                            selections={k: selections[k] for k in ("home", "away")},
                            bookmaker=str(book.get("name", "unknown")),
                        )
        return None

    def get_results(self, sport: Sport, start: date, end: date) -> list[MatchResult]:
        """Backfill finished singles matches as two-way settled results."""
        if sport is not Sport.TENNIS:
            raise ProviderError(
                f"APITennisProvider supports tennis only; got {sport.value}."
            )
        results: list[MatchResult] = []
        cur = start
        while cur <= end:
            raw = self._get("games", {"date": cur.isoformat()})
            for g in raw or []:
                status = (g.get("status") or "").lower()
                if status not in ("finished", "ft"):
                    continue
                players = g.get("players") or {}
                home = (players.get("home") or {}).get("name")
                away = (players.get("away") or {}).get("name")
                winner = (g.get("winner") or "").lower()  # "home" or "away"
                if home is None or away is None or winner not in ("home", "away"):
                    continue
                match = Match(
                    sport=Sport.TENNIS,
                    home=home,
                    away=away,
                    match_date=cur,
                    quality=DataQuality(effective_samples=0),
                )
                results.append(MatchResult(match, {"match_winner": winner}))
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
