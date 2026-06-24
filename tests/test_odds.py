import math

import pytest

from bet_assistant.config import ValueConfig
from bet_assistant.odds.vig import implied_probabilities, overround, remove_vig
from bet_assistant.odds.value import assess_value, expected_value
from bet_assistant.odds.kelly import kelly_fraction, recommended_stake
from bet_assistant.config import StakingConfig


def test_implied_probabilities_and_overround():
    odds = {"home": 2.0, "away": 2.0}
    imp = implied_probabilities(odds)
    assert imp == {"home": 0.5, "away": 0.5}
    assert overround(odds) == pytest.approx(1.0)


def test_remove_vig_sums_to_one():
    odds = {"home": 1.90, "draw": 3.50, "away": 4.20}
    fair = remove_vig(odds)
    assert sum(fair.values()) == pytest.approx(1.0)
    # Favourite keeps the largest share.
    assert fair["home"] > fair["away"] > 0


def test_remove_vig_rejects_bad_odds():
    with pytest.raises(ValueError):
        remove_vig({"x": 1.0})


def test_expected_value():
    # Fair coin at 2.10 => +5% EV.
    assert expected_value(0.5, 2.10) == pytest.approx(0.05)


def test_kelly_fraction():
    # p=0.6, odds=2.0 (b=1) => f* = (1*0.6 - 0.4)/1 = 0.2
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
    # No edge => no stake.
    assert kelly_fraction(0.5, 2.0) == 0.0


def test_recommended_stake_is_capped_and_fractional():
    cfg = StakingConfig(kelly_fraction=0.25, max_stake_fraction=0.02)
    rec = recommended_stake(0.7, 2.0, bankroll=1000.0, config=cfg)
    # Full kelly would be 0.4; quarter is 0.1; cap is 0.02.
    assert rec.full_kelly_fraction == pytest.approx(0.4)
    assert rec.fraction_of_bankroll == pytest.approx(0.02)
    assert rec.capped is True
    assert rec.stake == pytest.approx(20.0)


def test_assess_value_flags_only_real_edge():
    cfg = ValueConfig(min_edge_margin=0.02, min_expected_value=0.01,
                      min_effective_samples=10)
    odds = {"home": 2.0, "away": 2.0}  # vig-free 50/50
    # Model thinks home is 56% — a 6pt edge.
    out = assess_value("ml", odds, {"home": 0.56, "away": 0.44}, cfg,
                       effective_samples=50)
    by_sel = {a.selection: a for a in out}
    assert by_sel["home"].is_value is True
    assert by_sel["away"].is_value is False


def test_assess_value_insufficient_data_blocks_all():
    cfg = ValueConfig(min_effective_samples=30)
    odds = {"home": 2.0, "away": 2.0}
    out = assess_value("ml", odds, {"home": 0.9, "away": 0.1}, cfg,
                       effective_samples=5)
    assert all(not a.is_value for a in out)
    assert all(a.no_bet_reason for a in out)
