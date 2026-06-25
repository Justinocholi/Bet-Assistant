"""Tests for the reusable demo analysis that backs the web/serverless API."""

import pytest

from bet_assistant.demo import run_analysis, run_demo, supported_sports
from bet_assistant.language import find_forbidden


def test_supported_sports():
    assert set(supported_sports()) == {"football", "basketball", "tennis"}


@pytest.mark.parametrize("sport", ["football", "basketball", "tennis"])
def test_run_demo_shape_and_serialisable(sport):
    import json

    payload = run_demo(sport)
    # Round-trips as JSON (the serverless function ships it over the wire).
    json.dumps(payload)

    assert payload["sport"] == sport
    assert payload["error"] is None
    assert "notice" in payload and payload["notice"]
    assert "disclaimer" in payload
    assert isinstance(payload["recommendations"], list)
    assert isinstance(payload["no_bets"], list)
    assert payload["summary"]["n_value"] == len(payload["recommendations"])


@pytest.mark.parametrize("sport", ["football", "basketball", "tennis"])
def test_demo_output_carries_no_certainty_language(sport):
    payload = run_demo(sport)
    blobs = [payload["notice"], payload["disclaimer"]]
    for r in payload["recommendations"]:
        blobs.append(r["reasoning"])
    for n in payload["no_bets"]:
        blobs.append(n["reason"])
    for text in blobs:
        assert not find_forbidden(text), f"forbidden certainty term in: {text!r}"


def test_recommendations_have_uncertainty_band():
    payload = run_demo("basketball")
    for r in payload["recommendations"]:
        # A band, never a single false-precise number.
        assert r["band_low_pct"] <= r["model_prob_pct"] <= r["band_high_pct"]
        assert r["band_high_pct"] > r["band_low_pct"]
        # Edge is positive for anything flagged as value.
        assert r["edge_pts"] > 0


def test_unsupported_sport_raises():
    with pytest.raises(ValueError):
        run_demo("cricket")
    with pytest.raises(ValueError):
        run_analysis("cricket")


def test_run_analysis_falls_back_to_synthetic_without_key():
    payload = run_analysis("football")  # no api_key
    assert payload["source"] == "synthetic"
    assert "note" in payload
    assert payload["error"] is None


def test_run_analysis_live_football_uses_real_provider(monkeypatch):
    # Reuse the API-Football fake transport to drive a live run offline.
    from datetime import date
    from tests.test_apifootball import FakeTransport
    from bet_assistant.data import apifootball

    real_init = apifootball.APIFootballProvider.__init__

    def patched_init(self, api_key, league, season, base_url=apifootball._BASE_URL,
                     transport=None):
        real_init(self, api_key, league, season, base_url, transport=FakeTransport())

    monkeypatch.setattr(apifootball.APIFootballProvider, "__init__", patched_init)

    payload = run_analysis(
        "football", api_key="k", league=39, season=2023, on=date(2023, 8, 1)
    )
    assert payload["source"] == "api-football"
    assert payload["error"] is None
    # The fake returns one fully-formed fixture; it should be assessed (a value
    # bet or a no-bet), and the output must round-trip as JSON.
    import json
    json.dumps(payload)
    assert payload["summary"]["n_value"] + payload["summary"]["n_no_bet"] >= 1
