"""Shared model output type and the insufficient-data signal."""

from __future__ import annotations

from dataclasses import dataclass, field


class InsufficientModelData(RuntimeError):
    """Raised by a model when inputs are too thin to produce a usable estimate.

    The pipeline catches this and emits a "no bet — insufficient data" result
    rather than guessing.
    """


@dataclass
class ModelOutput:
    """Probabilities for a market plus an honest uncertainty band.

    ``probabilities`` maps selection -> probability (sums to ~1 across the
    market). ``confidence_half_width`` is a probability-space half-width that
    applies to the estimates (wider = less certain). ``effective_samples`` is
    how much data backs the estimate, used by the value gate.
    """

    market: str
    probabilities: dict[str, float]
    confidence_half_width: float
    effective_samples: int
    notes: list[str] = field(default_factory=list)

    def band_for(self, selection: str) -> tuple[float, float]:
        p = self.probabilities[selection]
        lo = max(0.0, p - self.confidence_half_width)
        hi = min(1.0, p + self.confidence_half_width)
        return lo, hi
