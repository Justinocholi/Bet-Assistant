"""Tests for the hard requirements: no certainty language, model gating,
bankroll stop-loss, self-exclusion, calibration, and graceful degradation.
"""

from datetime import date

import pytest

from bet_assistant.language import (
    CertaintyLanguageError,
    assert_uncertain,
    find_forbidden,
)
from bet_assistant.config import Config, ModelGate, BankrollConfig
from bet_assistant.data.providers import MockProvider, safe_get_fixtures, ProviderError
from bet_assistant.data.schema import Match, Sport, TeamForm, DataQuality
from bet_assistant.models.base import ModelOutput
from bet_assistant.pipeline import evaluate_market, Recommendation, NoBet
from bet_assistant.bankroll import (
    BankrollManager,
    ResponsibleGambling,
    SelfExclusionActive,
    StakingHalted,
)
from bet_assistant.backtest.calibration import calibration_error, reliability_plot


# -- forbidden language --------------------------------------------------

@pytest.mark.parametrize("bad", [
    "this is a guaranteed winner",
    "a sure thing tonight",
    "definitely the lock of the day",
    "100% safe, risk-free",
])
def test_forbidden_language_detected(bad):
    assert find_forbidden(bad)
    with pytest.raises(CertaintyLanguageError):
        assert_uncertain(bad)


def test_clean_text_passes():
    ok = "Model estimates 58% vs market 52%; moderate confidence, never certain."
    assert assert_uncertain(ok) == ok


def test_rendered_recommendation_has_no_certainty():
    match = Match(Sport.BASKETBALL, "H", "A", date(2025, 1, 1),
                  quality=DataQuality(50))
    output = ModelOutput("moneyline", {"home": 0.62, "away": 0.38},
                         confidence_half_width=0.05, effective_samples=50)
    config = Config(models=ModelGate(basketball_elo=True))
    results = evaluate_market(match, output, "moneyline",
                              {"home": 2.0, "away": 2.0}, 1000.0,
                              "basketball_elo", config)
    recs = [r for r in results if isinstance(r, Recommendation)]
    assert recs, "expected a value bet"
    for r in recs:
        # render() runs assert_uncertain internally; should not raise.
        assert "VALUE" in r.render()


# -- model gating --------------------------------------------------------

def test_unvalidated_model_cannot_flag_bets():
    match = Match(Sport.BASKETBALL, "H", "A", date(2025, 1, 1),
                  quality=DataQuality(50))
    output = ModelOutput("moneyline", {"home": 0.9, "away": 0.1},
                         confidence_half_width=0.05, effective_samples=50)
    # Default config: basketball_elo disabled.
    results = evaluate_market(match, output, "moneyline",
                              {"home": 2.0, "away": 2.0}, 1000.0,
                              "basketball_elo", Config())
    assert all(isinstance(r, NoBet) for r in results)
    assert all("not validated" in r.reason for r in results)


# -- bankroll & stop-loss ------------------------------------------------

def test_stop_loss_halts_staking():
    cfg = BankrollConfig(starting_bankroll=100.0, stop_loss_fraction=0.25)
    mgr = BankrollManager(cfg)
    # Lose enough to cross the 75.0 floor.
    rec = mgr.place_bet("ml", "home", stake=30.0, decimal_odds=2.0,
                        model_prob=0.5, today=date(2025, 1, 1))
    mgr.settle_bet(rec, won=False)  # bankroll now 70 < 75 floor
    allowed, reason = mgr.can_stake(today=date(2025, 1, 2))
    assert not allowed and "Stop-loss" in reason
    with pytest.raises(StakingHalted):
        mgr.place_bet("ml", "home", 10.0, 2.0, 0.5, today=date(2025, 1, 2))


def test_winning_bet_returns_stake_plus_winnings():
    cfg = BankrollConfig(starting_bankroll=100.0)
    mgr = BankrollManager(cfg)
    rec = mgr.place_bet("ml", "home", 10.0, 2.0, 0.6, today=date(2025, 1, 1))
    assert mgr.bankroll == pytest.approx(90.0)  # stake reserved
    mgr.settle_bet(rec, won=True)
    assert mgr.bankroll == pytest.approx(110.0)  # 90 + 10*2.0
    assert "ROI" in mgr.report()


def test_losing_streak_reported():
    cfg = BankrollConfig(starting_bankroll=1000.0)
    mgr = BankrollManager(cfg)
    for _ in range(3):
        rec = mgr.place_bet("ml", "home", 5.0, 2.0, 0.5, today=date(2025, 1, 1))
        mgr.settle_bet(rec, won=False)
    assert "Current losing streak: 3" in mgr.report()


# -- self-exclusion / cool-off ------------------------------------------

def test_self_exclusion_blocks_staking():
    rg = ResponsibleGambling()
    rg.cool_off(7, today=date(2025, 1, 1))
    assert not rg.is_active(today=date(2025, 1, 3))
    with pytest.raises(SelfExclusionActive):
        rg.assert_active(today=date(2025, 1, 3))
    # Period elapses.
    assert rg.is_active(today=date(2025, 1, 10))


def test_manager_respects_self_exclusion():
    cfg = BankrollConfig(starting_bankroll=100.0)
    rg = ResponsibleGambling()
    rg.self_exclude(30, today=date(2025, 1, 1))
    mgr = BankrollManager(cfg, responsible=rg)
    allowed, reason = mgr.can_stake(today=date(2025, 1, 5))
    assert not allowed and "Paused" in reason
    with pytest.raises(StakingHalted):
        mgr.place_bet("ml", "home", 5.0, 2.0, 0.5, today=date(2025, 1, 5))


# -- graceful degradation ------------------------------------------------

def test_provider_failure_degrades_without_crashing():
    provider = MockProvider(seed=1, fail_rate=1.0)
    matches, err = safe_get_fixtures(provider, Sport.FOOTBALL, date(2025, 1, 1))
    assert matches == []
    assert err and "unavailable" in err


def test_insufficient_data_yields_no_bet():
    # Thin data -> below the min_effective_samples gate -> no bet.
    match = Match(Sport.BASKETBALL, "H", "A", date(2025, 1, 1),
                  quality=DataQuality(3))
    output = ModelOutput("moneyline", {"home": 0.9, "away": 0.1},
                         confidence_half_width=0.2, effective_samples=3)
    config = Config(models=ModelGate(basketball_elo=True))
    results = evaluate_market(match, output, "moneyline",
                              {"home": 2.0, "away": 2.0}, 1000.0,
                              "basketball_elo", config)
    assert all(isinstance(r, NoBet) for r in results)
    assert any("Insufficient data" in r.reason for r in results)


# -- calibration ---------------------------------------------------------

def test_perfectly_calibrated_has_low_error():
    # Construct predictions where observed == predicted in each band.
    preds = []
    for p, n_win, n_total in [(0.2, 20, 100), (0.5, 50, 100), (0.8, 80, 100)]:
        preds += [(p, True)] * n_win + [(p, False)] * (n_total - n_win)
    assert calibration_error(preds) < 0.02
    assert "Reliability" in reliability_plot(preds)


def test_miscalibrated_has_high_error():
    # Model says 0.8 but they win only 30%.
    preds = [(0.8, True)] * 30 + [(0.8, False)] * 70
    assert calibration_error(preds) > 0.4
