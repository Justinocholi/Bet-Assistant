"""Tests for the reusable demo analysis that backs the web/serverless API."""

import pytest

from bet_assistant.demo import run_demo, supported_sports
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
