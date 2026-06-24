"""Bankroll manager: tracks every bet, enforces a hard stop-loss, reports
running ROI honestly — including losing streaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from ..config import BankrollConfig
from .responsible import ResponsibleGambling


class StakingHalted(RuntimeError):
    """Raised when a stake is attempted after the stop-loss has triggered."""


@dataclass
class BetRecord:
    market: str
    selection: str
    stake: float
    decimal_odds: float
    model_prob: float
    placed_on: date
    settled: bool = False
    won: Optional[bool] = None

    @property
    def profit(self) -> float:
        if not self.settled or self.won is None:
            return 0.0
        return self.stake * (self.decimal_odds - 1.0) if self.won else -self.stake


@dataclass
class BankrollManager:
    config: BankrollConfig
    responsible: ResponsibleGambling = field(default_factory=ResponsibleGambling)
    bankroll: float = field(init=False)
    starting: float = field(init=False)
    bets: list[BetRecord] = field(default_factory=list)
    _halted: bool = False

    def __post_init__(self) -> None:
        self.starting = self.config.starting_bankroll
        self.bankroll = self.config.starting_bankroll

    # -- guards ------------------------------------------------------------

    @property
    def stop_loss_floor(self) -> float:
        return self.starting * (1.0 - self.config.stop_loss_fraction)

    @property
    def stop_loss_triggered(self) -> bool:
        return self._halted or self.bankroll <= self.stop_loss_floor

    def can_stake(self, today: Optional[date] = None) -> tuple[bool, str]:
        """Return (allowed, reason). Checks self-exclusion then stop-loss."""
        if not self.responsible.is_active(today):
            return False, self.responsible.status(today)
        if self.stop_loss_triggered:
            self._halted = True
            return False, (
                f"Stop-loss reached: bankroll {self.bankroll:.2f} at/under floor "
                f"{self.stop_loss_floor:.2f}. Staking halted until you reset."
            )
        return True, "ok"

    # -- bet lifecycle -----------------------------------------------------

    def place_bet(
        self,
        market: str,
        selection: str,
        stake: float,
        decimal_odds: float,
        model_prob: float,
        placed_on: Optional[date] = None,
        today: Optional[date] = None,
    ) -> BetRecord:
        allowed, reason = self.can_stake(today)
        if not allowed:
            raise StakingHalted(reason)
        if stake <= 0:
            raise ValueError("stake must be positive")
        if stake > self.bankroll:
            raise StakingHalted("stake exceeds available bankroll")
        rec = BetRecord(
            market=market,
            selection=selection,
            stake=stake,
            decimal_odds=decimal_odds,
            model_prob=model_prob,
            placed_on=placed_on or today or date.today(),
        )
        self.bankroll -= stake  # reserve the stake
        self.bets.append(rec)
        return rec

    def settle_bet(self, rec: BetRecord, won: bool) -> None:
        if rec.settled:
            raise ValueError("bet already settled")
        rec.settled = True
        rec.won = won
        if won:
            self.bankroll += rec.stake * rec.decimal_odds  # return stake + winnings
        # if lost, stake was already deducted at placement
        if self.stop_loss_triggered:
            self._halted = True

    def reset_stop_loss(self) -> None:
        """Explicit, deliberate reset — never automatic."""
        self._halted = False

    # -- honest reporting --------------------------------------------------

    def report(self) -> str:
        settled = [b for b in self.bets if b.settled]
        staked = sum(b.stake for b in settled)
        profit = sum(b.profit for b in settled)
        wins = sum(1 for b in settled if b.won)
        roi = profit / staked if staked > 0 else 0.0
        streak = self._current_losing_streak(settled)
        lines = [
            f"Bankroll: {self.bankroll:.2f} (started {self.starting:.2f})",
            f"Settled bets: {len(settled)} | wins: {wins} | "
            f"hit rate: {(wins/len(settled)*100) if settled else 0:.1f}%",
            f"Total staked: {staked:.2f} | profit: {profit:+.2f} | "
            f"ROI: {roi*100:+.1f}%",
            f"Current losing streak: {streak}",
            f"Stop-loss floor: {self.stop_loss_floor:.2f} | "
            f"triggered: {self.stop_loss_triggered}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _current_losing_streak(settled: list[BetRecord]) -> int:
        streak = 0
        for b in reversed(settled):
            if b.won is False:
                streak += 1
            else:
                break
        return streak
