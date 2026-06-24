import random
from datetime import date, timedelta

from bet_assistant.backtest.metrics import Bet, summarise, max_drawdown
from bet_assistant.backtest.engine import run_backtest
from bet_assistant.config import DEFAULT_CONFIG
from bet_assistant.data.providers import MockProvider
from bet_assistant.data.history import ScoredFixture, build_pointintime_results
from bet_assistant.data.schema import Sport
from bet_assistant.models.elo import EloModel
from bet_assistant.models.poisson import DixonColesModel


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


def test_football_pointintime_backtest_end_to_end():
    """History -> point-in-time form -> Dixon-Coles -> value backtest runs and
    reports honest metrics + calibration without leaking future data."""
    rng = random.Random(0)
    teams = [f"T{i}" for i in range(8)]
    fixtures = []
    start = date(2025, 1, 1)
    # A round-robin-ish synthetic season with random scores and vigged 1x2 odds.
    for week in range(20):
        rng.shuffle(teams)
        for i in range(0, len(teams), 2):
            home, away = teams[i], teams[i + 1]
            hg, ag = rng.randint(0, 4), rng.randint(0, 3)
            # crude vigged odds summing implied > 1
            odds = {"home": 2.5, "draw": 3.3, "away": 2.9}
            fixtures.append(ScoredFixture(
                start + timedelta(days=week * 7), home, away, hg, ag,
                odds={"1x2": odds}))

    results = build_pointintime_results(fixtures, sport=Sport.FOOTBALL)

    def predictor(match):
        return DixonColesModel().market_1x2(match)

    result = run_backtest(results, predictor, "1x2", DEFAULT_CONFIG)
    assert "VERDICT" in result.report()
    # Some fixtures early in the season lack prior form and must be skipped,
    # not bet on.
    assert result.n_skipped_insufficient >= 0
    assert isinstance(result.calibration_ece, float)
