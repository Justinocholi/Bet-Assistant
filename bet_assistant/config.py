"""Central, tunable configuration.

Everything that controls how aggressive or conservative the tool is lives here
so it can be reviewed in one place. Defaults are deliberately conservative.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ValueConfig:
    """Controls when a bet is flagged as value."""

    # A bet is flagged only when model_prob exceeds the vig-free implied prob
    # by at least this absolute margin. This is a buffer against model error,
    # not a free parameter to chase more bets — raising it means *fewer*, more
    # confident flags.
    min_edge_margin: float = 0.02  # 2 percentage points of probability

    # Minimum expected value (per unit staked) to bother flagging.
    min_expected_value: float = 0.01  # 1%

    # If the model's effective sample size behind a probability is below this,
    # we declare "insufficient data" and never flag a bet.
    min_effective_samples: int = 30


@dataclass(frozen=True)
class StakingConfig:
    """Controls stake sizing. Fractional Kelly, hard-capped."""

    # Fraction of full Kelly to use. Quarter-Kelly is a common, conservative
    # choice that sharply reduces variance and drawdown.
    kelly_fraction: float = 0.25

    # Never stake more than this fraction of bankroll on a single bet,
    # regardless of what Kelly suggests.
    max_stake_fraction: float = 0.02  # 2% of bankroll

    # Below this stake fraction it isn't worth placing — treat as no bet.
    min_stake_fraction: float = 0.001  # 0.1%


@dataclass(frozen=True)
class BankrollConfig:
    """Bankroll management and hard stop-loss."""

    starting_bankroll: float = 1000.0

    # Hard stop: if cumulative losses reach this fraction of the *starting*
    # bankroll, staking halts until the user explicitly resets.
    stop_loss_fraction: float = 0.25  # stop after losing 25% of the bankroll


@dataclass(frozen=True)
class ModelGate:
    """Whether a model is allowed to flag live bets.

    Every model starts disabled. It may only be enabled after the backtest
    harness confirms it beats the vig-free closing line on held-out seasons.
    """

    football_poisson: bool = False
    basketball_elo: bool = False
    tennis_glicko: bool = False
    logistic_binary: bool = False


@dataclass(frozen=True)
class Config:
    value: ValueConfig = field(default_factory=ValueConfig)
    staking: StakingConfig = field(default_factory=StakingConfig)
    bankroll: BankrollConfig = field(default_factory=BankrollConfig)
    models: ModelGate = field(default_factory=ModelGate)


DEFAULT_CONFIG = Config()
