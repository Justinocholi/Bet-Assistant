"""Calibration / reliability analysis.

The central honesty check: when the model says 60%, do those bets win ~60% of
the time? We bin predictions by probability and compare the mean predicted
probability in each bin to the observed win frequency. A well-calibrated model
sits on the diagonal.

We also compute the Expected Calibration Error (ECE) as a single summary, and
render an ASCII reliability plot so it shows up in a terminal with no plotting
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float

    @property
    def gap(self) -> float:
        return self.observed_rate - self.mean_predicted


def calibration_table(
    predictions: list[tuple[float, bool]], n_bins: int = 10
) -> list[CalibrationBin]:
    """Bin (predicted_prob, won) pairs into ``n_bins`` equal-width buckets."""
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, won in predictions:
        idx = min(n_bins - 1, max(0, int(p * n_bins)))
        bins[idx].append((p, won))

    table: list[CalibrationBin] = []
    for i, bucket in enumerate(bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if not bucket:
            table.append(CalibrationBin(lo, hi, 0, float("nan"), float("nan")))
            continue
        mean_pred = sum(p for p, _ in bucket) / len(bucket)
        obs = sum(1 for _, w in bucket if w) / len(bucket)
        table.append(CalibrationBin(lo, hi, len(bucket), mean_pred, obs))
    return table


def calibration_error(predictions: list[tuple[float, bool]], n_bins: int = 10) -> float:
    """Expected Calibration Error: count-weighted mean |observed - predicted|."""
    table = calibration_table(predictions, n_bins)
    total = sum(b.count for b in table)
    if total == 0:
        return float("nan")
    return sum(b.count * abs(b.gap) for b in table if b.count) / total


def reliability_plot(predictions: list[tuple[float, bool]], n_bins: int = 10) -> str:
    """A compact ASCII reliability plot. 'p' = predicted, 'o' = observed."""
    table = calibration_table(predictions, n_bins)
    width = 40
    lines = ["Reliability (predicted vs observed win rate):"]
    for b in table:
        if not b.count:
            lines.append(f"  [{b.lower:.1f}-{b.upper:.1f})  (no bets)")
            continue
        pp = int(round(b.mean_predicted * width))
        op = int(round(b.observed_rate * width))
        row = [" "] * (width + 1)
        row[min(width, pp)] = "p"
        row[min(width, op)] = "o" if op != pp else "X"
        lines.append(
            f"  [{b.lower:.1f}-{b.upper:.1f}) n={b.count:<4d} |{''.join(row)}| "
            f"pred {b.mean_predicted:.0%} obs {b.observed_rate:.0%}"
        )
    lines.append(f"  ECE = {calibration_error(predictions, n_bins):.3f} "
                 "(lower is better; 'X' = predicted≈observed)")
    return "\n".join(lines)
