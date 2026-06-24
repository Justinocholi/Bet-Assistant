"""Offline tests for the basketball and tennis adapters via fake transports."""

from datetime import date

import pytest

from bet_assistant.data.apibasketball import APIBasketballProvider
from bet_assistant.data.apitennis import APITennisProvider
from bet_assistant.data.http import HttpError
from bet_assistant.data.providers import ProviderError
from bet_assistant.data.schema import Sport
from bet_assistant.models.elo import EloModel
from bet_assistant.models.glicko import GlickoModel


# -- basketball ----------------------------------------------------------

class FakeBasketball:
    def __init__(self, fail_paths=()):
        self.fail_paths = set(fail_paths)

    def __call__(self, url, headers):
        for p in self.fail_paths:
            if p in url:
                raise HttpError(f"fail {p}")
        if "/games" in url:
            return {"errors": [], "response": [{
                "id": 55,
                "status": {"short": "FT"},
                "teams": {"home": {"id": 1, "name": "Lakers"},
                          "away": {"id": 2, "name": "Celtics"}},
                "scores": {"home": {"total": 110}, "away": {"total": 102}},
            }]}
        if "/teams/statistics" in url:
            return {"errors": [], "response": {
                "games": {"played": {"all": 40}},
                "points": {
                    "for": {"average": {"home": "112.5", "away": "108.1"}},
                    "against": {"average": {"home": "104.2", "away": "107.0"}},
                },
            }}
        if "/odds" in url:
            return {"errors": [], "response": [{
                "bookmakers": [{"name": "BK", "bets": [{
                    "id": 2, "name": "Home/Away",
                    "values": [{"value": "Home", "odd": "1.80"},
                               {"value": "Away", "odd": "2.05"}],
                }]}]
            }]}
        raise AssertionError(url)


def _bb(transport):
    return APIBasketballProvider("k", league=12, season="2023-2024",
                                 transport=transport)


def test_basketball_fixture_form_and_odds():
    m = _bb(FakeBasketball()).get_fixtures(Sport.BASKETBALL, date(2024, 1, 1))[0]
    assert m.home == "Lakers" and m.away == "Celtics"
    assert m.home_form.matches_played == 40
    assert m.home_form.goals_for_home == pytest.approx(112.5)
    line = m.odds_for("moneyline")
    assert set(line.selections) == {"home", "away"}
    assert line.selections["home"] == pytest.approx(1.80)


def test_basketball_results_train_elo():
    results = _bb(FakeBasketball()).get_results(
        Sport.BASKETBALL, date(2024, 1, 1), date(2024, 1, 1))
    assert results and results[0].outcomes["moneyline"] == "home"
    elo = EloModel().train(results)
    out = elo.market_moneyline("Lakers", "Celtics")
    assert out.probabilities["home"] > 0.5  # Lakers beat Celtics in the data


def test_basketball_odds_failure_flagged_not_faked():
    m = _bb(FakeBasketball(fail_paths=["/odds"])).get_fixtures(
        Sport.BASKETBALL, date(2024, 1, 1))[0]
    assert m.odds_for("moneyline") is None
    assert "odds" in m.quality.missing_fields


def test_basketball_rejects_other_sport():
    with pytest.raises(ProviderError):
        _bb(FakeBasketball()).get_fixtures(Sport.FOOTBALL, date(2024, 1, 1))


# -- tennis --------------------------------------------------------------

class FakeTennis:
    def __init__(self, fail_paths=()):
        self.fail_paths = set(fail_paths)

    def __call__(self, url, headers):
        for p in self.fail_paths:
            if p in url:
                raise HttpError(f"fail {p}")
        if "/players/statistics" in url:
            return {"errors": [], "response": {
                "matches_played": 30,
                "surface_win_rate": 0.62,
                "form": "WWLWW",
                "injured": False,
            }}
        if "/games" in url:
            return {"errors": [], "response": [{
                "id": 7,
                "surface": "Hard",
                "status": "finished",
                "winner": "home",
                "players": {"home": {"id": 100, "name": "Player A"},
                            "away": {"id": 200, "name": "Player B"}},
            }]}
        if "/odds" in url:
            return {"errors": [], "response": [{
                "bookmakers": [{"name": "BK", "bets": [{
                    "values": [{"value": "Home", "odd": "1.65"},
                               {"value": "Away", "odd": "2.30"}],
                }]}]
            }]}
        raise AssertionError(url)


def _tn(transport):
    return APITennisProvider("k", base_url="https://tennis.example/v1",
                             season=2024, transport=transport)


def test_tennis_requires_base_url():
    with pytest.raises(ValueError):
        APITennisProvider("k", base_url="", season=2024)


def test_tennis_fixture_form_and_odds():
    m = _tn(FakeTennis()).get_fixtures(Sport.TENNIS, date(2024, 1, 1))[0]
    assert m.home == "Player A" and m.away == "Player B"
    assert m.home_player.matches_played == 30
    assert m.home_player.win_rate_surface == pytest.approx(0.62)
    line = m.odds_for("match_winner")
    assert line.selections["home"] == pytest.approx(1.65)


def test_tennis_results_train_glicko():
    results = _tn(FakeTennis()).get_results(
        Sport.TENNIS, date(2024, 1, 1), date(2024, 1, 1))
    assert results[0].outcomes["match_winner"] == "home"
    g = GlickoModel().train(results)
    out = g.market_match_winner("Player A", "Player B")
    assert out.probabilities["home"] > 0.5


def test_tennis_form_failure_flagged():
    m = _tn(FakeTennis(fail_paths=["/players/statistics"])).get_fixtures(
        Sport.TENNIS, date(2024, 1, 1))[0]
    assert m.home_player is None
    assert "home_player" in m.quality.missing_fields
