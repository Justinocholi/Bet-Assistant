from datetime import date

from bet_assistant.backtest.metrics import Bet, summarise, max_drawdown
from bet_assistant.backtest.engine import run_backtest
from bet_assistant.config import DEFAULT_CONFIG
from bet_assistant.data.providers import MockProvider
from bet_assistant.data.schema import Sport
from bet_assistant.models.elo import EloModel


def test_max_drawdown_basic():
    # Win then two losses on a flat-staked book.
    bets = [
        Bet(0.5, 2.0, 100, won=True),   # +100 -> 1100
        Bet(0.5, 2.0, 100, won=False),  # -100 -> 1000
        Bet(0.5, 2.0, 100, won=False),  # -100 -> 900
    ]
    dd = max_drawdown(bets, starting_bankroll=1000.0)
    # Peak 1100, trough 900 => 200/1100.
    assert dd == round(200 / 1100, 10) or abs(dd - 200 / 1100) < 1e-9


def test_clv_detected_when_model_beats_closing():
    bets = [
        Bet(0.60, 2.0, 10, won=True, closing_vig_free_prob=0.50),
        Bet(0.55, 2.1, 10, won=False, closing_vig_free_prob=0.48),
    ]
    m = summarise(bets, 1000.0)
    assert m.beats_closing_line is True
    assert m.avg_clv > 0


def test_backtest_runs_and_reports():
    provider = MockProvider(seed=3)
    train = provider.get_results(Sport.BASKETBALL, date(2025, 1, 1), date(2025, 1, 20))
    test = provider.get_results(Sport.BASKETBALL, date(2025, 1, 21), date(2025, 2, 10))
    elo = EloModel().train(train)

    def predictor(match):
        return elo.market_moneyline(match.home, match.away)

    result = run_backtest(test, predictor, "moneyline", DEFAULT_CONFIG)
    # The report renders and the verdict gate is computed.
    assert "VERDICT" in result.report()
    assert isinstance(result.model_should_be_enabled, bool)
    # On synthetic fair odds, there is no real edge, so it should not be
    # auto-enabled (this is the safety property we care about).
    assert result.model_should_be_enabled is False
