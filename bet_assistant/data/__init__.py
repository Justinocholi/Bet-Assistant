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
]
