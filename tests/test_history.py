from datetime import date

import pytest

from bet_assistant.data.history import (
    ScoredFixture,
    outcomes_from_score,
    build_pointintime_results,
)
from bet_assistant.data.schema import Sport


def test_outcomes_from_score():
    o = outcomes_from_score(2, 1)
    assert o["1x2"] == "home"
    assert o["over_under_2.5"] == "over"
    assert o["btts"] == "yes"

    o = outcomes_from_score(0, 0)
    assert o["1x2"] == "draw"
    assert o["over_under_2.5"] == "under"
    assert o["btts"] == "no"

    assert outcomes_from_score(0, 3)["1x2"] == "away"


def test_pointintime_form_has_no_lookahead():
    # Team A plays three home games, scoring 1, then 2, then 3.
    fixtures = [
        ScoredFixture(date(2025, 1, 1), "A", "B", 1, 0),
        ScoredFixture(date(2025, 1, 8), "A", "C", 2, 0),
        ScoredFixture(date(2025, 1, 15), "A", "D", 3, 0),
    ]
    results = build_pointintime_results(fixtures, sport=Sport.FOOTBALL)

    # First fixture: A has no prior data -> no form, zero effective samples.
    first = results[0].match
    assert first.home_form is None
    assert first.quality.effective_samples == 0

    # Third fixture: A's home scoring avg should reflect ONLY the first two
    # games (1 and 2 -> avg 1.5), never the current (3rd) game.
    third = results[2].match
    assert third.home_form is not None
    assert third.home_form.goals_for_home == pytest.approx(1.5)
    assert third.home_form.matches_played == 2


def test_pointintime_rest_days_computed():
    fixtures = [
        ScoredFixture(date(2025, 1, 1), "A", "B", 1, 1),
        ScoredFixture(date(2025, 1, 6), "A", "C", 0, 0),  # 5 days later
    ]
    results = build_pointintime_results(fixtures)
    second = results[1].match
    assert second.home_form.rest_days == 5


def test_pointintime_attaches_odds_when_present():
    fixtures = [
        ScoredFixture(date(2025, 1, 1), "A", "B", 1, 0,
                      odds={"1x2": {"home": 2.0, "draw": 3.4, "away": 3.6}}),
    ]
    results = build_pointintime_results(fixtures)
    line = results[0].match.odds_for("1x2")
    assert line is not None and line.is_closing_line is True
