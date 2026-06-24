"""Backtesting and calibration. A model must pass here before it can flag bets."""

from .metrics import (
    BacktestMetrics,
    max_drawdown,
    roi_and_hit_rate,
    beats_closing_line,
)
from .calibration import CalibrationBin, calibration_table, calibration_error
from .engine import BacktestResult, run_backtest

__all__ = [
    "BacktestMetrics",
    "max_drawdown",
    "roi_and_hit_rate",
    "beats_closing_line",
    "CalibrationBin",
    "calibration_table",
    "calibration_error",
    "BacktestResult",
    "run_backtest",
]
