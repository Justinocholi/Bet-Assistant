"""Odds handling: vig removal, value/EV detection, fractional Kelly staking."""

from .vig import implied_probabilities, remove_vig, overround
from .value import ValueAssessment, assess_value, expected_value
from .kelly import kelly_fraction, recommended_stake, StakeRecommendation

__all__ = [
    "implied_probabilities",
    "remove_vig",
    "overround",
    "ValueAssessment",
    "assess_value",
    "expected_value",
    "kelly_fraction",
    "recommended_stake",
    "StakeRecommendation",
]
