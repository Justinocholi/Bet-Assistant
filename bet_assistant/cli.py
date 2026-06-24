"""Command-line demo entry point.

    python -m bet_assistant.cli demo      end-to-end demo on synthetic data
    python -m bet_assistant.cli notice    print the responsible-gambling notice
    python -m bet_assistant.cli backtest  backtest the Elo model and show calibration

Everything runs offline against the deterministic MockProvider — no API key.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from .config import Config, ModelGate, DEFAULT_CONFIG
from .data.providers import MockProvider, safe_get_fixtures
from .data.schema import Sport
from .models.elo import EloModel
from .pipeline import evaluate_market
from .bankroll import BankrollManager, responsible_gambling_notice


def _banner() -> str:
    line = "=" * 72
    return f"{line}\n{responsible_gambling_notice()}\n{line}"


def cmd_notice() -> int:
    print(_banner())
    return 0


def cmd_demo() -> int:
    print(_banner())
    provider = MockProvider(seed=42, fail_rate=0.0, thin_rate=0.25)

    # Train Elo on a season of history so it has ratings to work with.
    history = provider.get_results(
        Sport.BASKETBALL, date(2025, 1, 1), date(2025, 3, 1)
    )
    elo = EloModel().train(history)

    # Enable the basketball Elo model for the demo *as if* it had passed
    # validation. In production this flag is only set after a successful
    # backtest (see `backtest` command).
    config = Config(models=ModelGate(basketball_elo=True))

    today = date(2025, 3, 2)
    fixtures, err = safe_get_fixtures(provider, Sport.BASKETBALL, today)
    if err:
        print(f"\n[degraded] {err}\n(The tool surfaces this instead of guessing.)")
        return 0

    bankroll = config.bankroll.starting_bankroll
    print(f"\nAnalysing {len(fixtures)} basketball fixtures on {today} "
          f"(bankroll {bankroll:.0f}):\n")

    n_value = 0
    for match in fixtures:
        line = match.odds_for("moneyline")
        if line is None:
            continue
        try:
            output = elo.market_moneyline(match.home, match.away)
        except Exception as exc:
            print(f"  {match.home} vs {match.away}: no model output ({exc}).")
            continue

        results = evaluate_market(
            match=match,
            model_output=output,
            market="moneyline",
            decimal_odds=line.selections,
            bankroll=bankroll,
            model_key="basketball_elo",
            config=config,
        )
        for r in results:
            from .pipeline import Recommendation
            if isinstance(r, Recommendation):
                n_value += 1
                print(r.render())
                print()

    if n_value == 0:
        print("  No value bets found — that is a normal and frequent outcome.")
    print(f"\nFlagged {n_value} value bet(s). Everything else is 'no bet'.")
    return 0


def cmd_backtest() -> int:
    print(_banner())
    from .backtest.engine import run_backtest

    provider = MockProvider(seed=11)
    train = provider.get_results(Sport.BASKETBALL, date(2025, 1, 1), date(2025, 2, 1))
    test = provider.get_results(Sport.BASKETBALL, date(2025, 2, 2), date(2025, 3, 15))

    elo = EloModel().train(train)

    def predictor(match):
        return elo.market_moneyline(match.home, match.away)

    result = run_backtest(test, predictor, market="moneyline", config=DEFAULT_CONFIG)
    print("\nBacktest of basketball Elo (held-out period):\n")
    print(result.report())
    print(
        "\nNote: against the MockProvider, odds are sampled from the same fair "
        "probabilities the model can't see, so there is no real edge to find — "
        "the model correctly fails to beat the closing line and stays disabled. "
        "That is the system working as intended."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "demo"
    if cmd == "notice":
        return cmd_notice()
    if cmd == "demo":
        return cmd_demo()
    if cmd == "backtest":
        return cmd_backtest()
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
