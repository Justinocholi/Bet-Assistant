"""Offline tests for the API-Football adapter using a fake HTTP transport.

These verify the mapping from API-Football JSON to the tool's schema, plus the
two failure contracts: transport failures become ProviderError, and thin/absent
data is recorded in ``missing_fields`` rather than fabricated.
"""

from datetime import date

import pytest

from bet_assistant.data.apifootball import APIFootballProvider
from bet_assistant.data.http import HttpError
from bet_assistant.data.providers import ProviderError, safe_get_fixtures
from bet_assistant.data.schema import Sport
from bet_assistant.models.poisson import DixonColesModel


def _stats_response(form="WWDLW"):
    return {
        "errors": [],
        "response": {
            "form": form,
            "fixtures": {"played": {"home": 9, "away": 9, "total": 18}},
            "goals": {
                "for": {"average": {"home": "1.8", "away": "1.3", "total": "1.55"}},
                "against": {"average": {"home": "0.9", "away": "1.2", "total": "1.05"}},
            },
        },
    }


def _fixtures_response():
    return {
        "errors": [],
        "response": [
            {
                "fixture": {"id": 1001},
                "teams": {
                    "home": {"id": 33, "name": "Manchester United"},
                    "away": {"id": 40, "name": "Liverpool"},
                },
            }
        ],
    }


def _odds_response():
    return {
        "errors": [],
        "response": [
            {
                "bookmakers": [
                    {
                        "name": "TestBook",
                        "bets": [
                            {
                                "id": 1,
                                "name": "Match Winner",
                                "values": [
                                    {"value": "Home", "odd": "2.10"},
                                    {"value": "Draw", "odd": "3.40"},
                                    {"value": "Away", "odd": "3.60"},
                                ],
                            }
                        ],
                    }
                ]
            }
        ],
    }


def _h2h_response():
    return {
        "errors": [],
        "response": [
            {"teams": {"home": {"winner": True}, "away": {"winner": False}}},
            {"teams": {"home": {"winner": False}, "away": {"winner": False}}},
        ],
    }


class FakeTransport:
    """Routes fake JSON by the path embedded in the URL. Records calls."""

    def __init__(self, *, fail_paths=()):
        self.fail_paths = set(fail_paths)
        self.calls = []

    def __call__(self, url, headers):
        self.calls.append(url)
        for path in self.fail_paths:
            if path in url:
                raise HttpError(f"simulated failure for {path}")
        if "/fixtures/headtohead" in url:
            return _h2h_response()
        if "/fixtures" in url:
            return _fixtures_response()
        if "/teams/statistics" in url:
            return _stats_response()
        if "/odds" in url:
            return _odds_response()
        raise AssertionError(f"unexpected url {url}")


def _provider(transport):
    return APIFootballProvider(
        api_key="k", league=39, season=2023, transport=transport
    )


def test_requires_api_key():
    with pytest.raises(ValueError):
        APIFootballProvider(api_key="", league=39, season=2023)


def test_maps_fixture_form_and_odds():
    provider = _provider(FakeTransport())
    matches = provider.get_fixtures(Sport.FOOTBALL, date(2023, 8, 1))
    assert len(matches) == 1
    m = matches[0]
    assert m.home == "Manchester United" and m.away == "Liverpool"
    assert m.home_form and m.home_form.matches_played == 18
    assert m.home_form.goals_for_home == pytest.approx(1.8)
    assert m.quality.effective_samples == 18
    assert not m.quality.missing_fields

    line = m.odds_for("1x2")
    assert line is not None
    assert set(line.selections) == {"home", "draw", "away"}
    assert line.selections["home"] == pytest.approx(2.10)

    # The mapped match feeds the Dixon-Coles model cleanly.
    out = DixonColesModel().market_1x2(m)
    assert sum(out.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_h2h_mapped():
    provider = _provider(FakeTransport())
    m = provider.get_fixtures(Sport.FOOTBALL, date(2023, 8, 1))[0]
    assert m.head_to_head is not None
    assert m.head_to_head.home_or_first_wins == 1
    assert m.head_to_head.draws == 1


def test_fixtures_endpoint_failure_is_provider_error():
    provider = _provider(FakeTransport(fail_paths=["/fixtures?"]))
    with pytest.raises(ProviderError):
        provider.get_fixtures(Sport.FOOTBALL, date(2023, 8, 1))


def test_missing_odds_recorded_not_faked():
    provider = _provider(FakeTransport(fail_paths=["/odds"]))
    m = provider.get_fixtures(Sport.FOOTBALL, date(2023, 8, 1))[0]
    assert m.odds_for("1x2") is None
    assert "odds" in m.quality.missing_fields


def test_missing_form_recorded_not_faked():
    provider = _provider(FakeTransport(fail_paths=["/teams/statistics"]))
    m = provider.get_fixtures(Sport.FOOTBALL, date(2023, 8, 1))[0]
    assert m.home_form is None
    assert "home_form" in m.quality.missing_fields
    assert m.quality.effective_samples == 0


def test_non_football_sport_rejected():
    provider = _provider(FakeTransport())
    with pytest.raises(ProviderError):
        provider.get_fixtures(Sport.TENNIS, date(2023, 8, 1))


def test_safe_get_fixtures_degrades_on_failure():
    provider = _provider(FakeTransport(fail_paths=["/fixtures?"]))
    matches, err = safe_get_fixtures(provider, Sport.FOOTBALL, date(2023, 8, 1))
    assert matches == []
    assert err and "unavailable" in err
