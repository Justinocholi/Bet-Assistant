"""Data ingestion layer: schema, provider interface, and a safe mock provider."""

from .schema import (
    DataQuality,
    Match,
    OddsLine,
    TeamForm,
    PlayerForm,
    HeadToHead,
)
from .providers import (
    DataProvider,
    ProviderError,
    InsufficientDataError,
    MockProvider,
    safe_get_fixtures,
)
from .apifootball import APIFootballProvider
from .apibasketball import APIBasketballProvider
from .apitennis import APITennisProvider
from .history import (
    ScoredFixture,
    outcomes_from_score,
    build_pointintime_results,
)

__all__ = [
    "DataQuality",
    "Match",
    "OddsLine",
    "TeamForm",
    "PlayerForm",
    "HeadToHead",
    "DataProvider",
    "ProviderError",
    "InsufficientDataError",
    "MockProvider",
    "safe_get_fixtures",
    "APIFootballProvider",
    "APIBasketballProvider",
    "APITennisProvider",
    "ScoredFixture",
    "outcomes_from_score",
    "build_pointintime_results",
]
