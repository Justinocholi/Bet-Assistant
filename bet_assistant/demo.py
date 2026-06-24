"""Reusable demo analysis that returns structured, JSON-serialisable results.

Both the CLI and the web/serverless API call this so there is a single source
of truth for how a demo analysis is produced. It runs entirely offline against
the deterministic ``MockProvider`` — no API key, safe to host publicly.

IMPORTANT: this is a *demonstration* on synthetic fixtures. The models are
enabled here only to show the end-to-end output. On real money, a model is
enabled only after it passes the backtest harness (beats the vig-free closing
line). The payload says so explicitly, and the responsible-gambling notice is
always included.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from typing import Callable

from .bankroll import responsible_gambling_notice
from .config import Config, ModelGate
from .data.providers import MockProvider, safe_get_fixtures
from .data.schema import Match, Sport
from .models.base import InsufficientModelData, ModelOutput
from .models.elo import EloModel
from .models.glicko import GlickoModel
from .models.poisson import DixonColesModel
from .pipeline import NoBet, Recommendation, evaluate_market

# A sport's demo wiring: which market/model-gate to use and how to get a
# ModelOutput for a match (training already applied via the closure).
_SUPPORTED = ("football", "basketball", "tennis")


def supported_sports() -> tuple[str, ...]:
    return _SUPPORTED


def _build_predictor(sport: Sport, provider: MockProvider):
    """Return (market, model_key, predictor) for the sport, training as needed."""
    if sport is Sport.FOOTBALL:
        model = DixonColesModel()
        return "1x2", "football_poisson", (lambda m: model.market_1x2(m))

    if sport is Sport.BASKETBALL:
        history = provider.get_results(sport, date(2025, 1, 1), date(2025, 3, 1))
        elo = EloModel().train(history)
        return (
            "moneyline",
            "basketball_elo",
            (lambda m: elo.market_moneyline(m.home, m.away)),
        )

    if sport is Sport.TENNIS:
        history = provider.get_results(sport, date(2025, 1, 1), date(2025, 3, 1))
        glicko = GlickoModel().train(history)
        return (
            "moneyline",
            "tennis_glicko",
            (lambda m: glicko.market_match_winner(m.home, m.away)),
        )

    raise ValueError(f"unsupported sport: {sport}")


def _enabled_config(model_key: str) -> Config:
    """Enable just the one demo model, as if it had passed validation."""
    gate_kwargs = {model_key: True}
    return Config(models=ModelGate(**gate_kwargs))


def run_demo(sport_name: str, *, seed: int = 42, on: date | None = None) -> dict:
    """Run a demo analysis for one sport and return a JSON-serialisable dict."""
    name = (sport_name or "basketball").lower()
    if name not in _SUPPORTED:
        raise ValueError(
            f"unsupported sport {name!r}; choose one of {', '.join(_SUPPORTED)}"
        )
    sport = Sport(name)
    on = on or date(2025, 3, 2)

    provider = MockProvider(seed=seed, fail_rate=0.0, thin_rate=0.25)
    market, model_key, predictor = _build_predictor(sport, provider)
    config = _enabled_config(model_key)
    bankroll = config.bankroll.starting_bankroll

    fixtures, error = safe_get_fixtures(provider, sport, on)
    payload: dict = {
        "sport": name,
        "date": on.isoformat(),
        "bankroll": bankroll,
        "market": market,
        "notice": responsible_gambling_notice(),
        "disclaimer": (
            "Demonstration on synthetic data. Models are enabled here only to "
            "illustrate the output; on real markets a model is enabled only "
            "after it beats the vig-free closing line in backtest. No outcome "
            "is ever assured."
        ),
        "recommendations": [],
        "no_bets": [],
        "error": error,
    }
    if error:
        # Graceful degradation: surface the failure, never fabricate fixtures.
        return payload

    for match in fixtures:
        line = match.odds_for(market)
        if line is None:
            continue
        try:
            output: ModelOutput = predictor(match)
        except InsufficientModelData as exc:
            payload["no_bets"].append(
                {
                    "match": f"{match.home} vs {match.away}",
                    "market": market,
                    "selection": "—",
                    "reason": f"Insufficient data for the model: {exc}",
                }
            )
            continue

        results = evaluate_market(
            match=match,
            model_output=output,
            market=market,
            decimal_odds=line.selections,
            bankroll=bankroll,
            model_key=model_key,
            config=config,
        )
        for r in results:
            if isinstance(r, Recommendation):
                row = asdict(r)
                row["match"] = f"{match.home} vs {match.away}"
                # Pre-format the values the UI needs, keeping raw numbers too.
                row["model_prob_pct"] = round(r.model_prob * 100, 1)
                row["vig_free_pct"] = round(r.vig_free_prob * 100, 1)
                row["edge_pts"] = round(r.edge * 100, 1)
                row["ev_pct"] = round(r.expected_value * 100, 1)
                row["band_low_pct"] = round(r.confidence_low * 100, 1)
                row["band_high_pct"] = round(r.confidence_high * 100, 1)
                row["stake_pct"] = round(r.stake_fraction * 100, 2)
                payload["recommendations"].append(row)
            elif isinstance(r, NoBet):
                payload["no_bets"].append(
                    {
                        "match": f"{match.home} vs {match.away}",
                        "market": r.market,
                        "selection": r.selection,
                        "reason": r.reason,
                    }
                )

    payload["summary"] = {
        "n_value": len(payload["recommendations"]),
        "n_no_bet": len(payload["no_bets"]),
        "message": (
            f"Flagged {len(payload['recommendations'])} value bet(s). "
            "Everything else is 'no bet' — a normal and frequent outcome."
        ),
    }
    return payload
