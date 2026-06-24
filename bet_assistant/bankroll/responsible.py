"""Responsible-gambling controls: persistent notice, self-exclusion, cool-off.

These are not optional decorations. The tool refuses to recommend or record any
stake while a self-exclusion or cool-off period is active.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from .. import RESPONSIBLE_GAMBLING_NOTICE


def responsible_gambling_notice() -> str:
    """The persistent notice shown alongside every batch of output."""
    return RESPONSIBLE_GAMBLING_NOTICE


class SelfExclusionActive(RuntimeError):
    """Raised on any staking attempt while excluded or in a cool-off period."""


@dataclass
class ResponsibleGambling:
    """Tracks a user's self-exclusion / cool-off state.

    ``excluded_until`` is the date staking may resume. ``None`` means active.
    Self-exclusion and cool-off use the same mechanism; cool-off is just a
    shorter, user-chosen duration.
    """

    excluded_until: Optional[date] = None

    def cool_off(self, days: int, today: Optional[date] = None) -> date:
        """Start a short cool-off period (e.g. 1, 7, 30 days)."""
        if days <= 0:
            raise ValueError("cool-off must be at least 1 day")
        today = today or date.today()
        self.excluded_until = today + timedelta(days=days)
        return self.excluded_until

    def self_exclude(self, days: int = 180, today: Optional[date] = None) -> date:
        """Start a longer self-exclusion (default 6 months)."""
        return self.cool_off(days, today)

    def is_active(self, today: Optional[date] = None) -> bool:
        """True if staking is currently allowed (not excluded)."""
        if self.excluded_until is None:
            return True
        today = today or date.today()
        if today >= self.excluded_until:
            self.excluded_until = None  # period elapsed; auto-clear
            return True
        return False

    def assert_active(self, today: Optional[date] = None) -> None:
        if not self.is_active(today):
            raise SelfExclusionActive(
                f"Staking is paused until {self.excluded_until.isoformat()}. "
                "This is a self-imposed break and is being respected. "
                "Support is available (e.g. US: 1-800-GAMBLER)."
            )

    def status(self, today: Optional[date] = None) -> str:
        if self.is_active(today):
            return "Active — no exclusion or cool-off in place."
        return f"Paused until {self.excluded_until.isoformat()} (self-exclusion/cool-off)."
