"""Small, dependency-free numerical helpers (no numpy/scipy)."""

from __future__ import annotations

import math


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for X ~ Poisson(lam)."""
    if lam < 0:
        raise ValueError("lambda must be >= 0")
    if k < 0:
        return 0.0
    return math.exp(-lam) * lam**k / math.factorial(k)


def sigmoid(x: float) -> float:
    # Numerically stable logistic function.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wilson_half_width(p: float, n: int, z: float = 1.96) -> float:
    """Half-width of a Wilson-style confidence interval for a proportion.

    Used to attach an honest uncertainty band to a probability estimate that is
    backed by ``n`` effective observations. More data -> tighter band.
    """
    if n <= 0:
        return 0.5  # maximal uncertainty
    p = clamp(p, 0.0, 1.0)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # Return the symmetric half-width around the (shrunk) centre, clamped to [0,0.5].
    return clamp(margin, 0.0, 0.5)
