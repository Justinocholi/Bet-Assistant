"""Reusable analysis that returns structured, JSON-serialisable results.

Both the CLI and the web/serverless API call this so there is a single source
of truth. Two entry points:

* ``run_demo(sport)`` — runs entirely offline against the deterministic
  ``MockProvider``. No API key, safe to host publicly.
* ``run_analysis(sport, api_key=...)`` — ingests **real** data via
  ``APIFootballProvider`` when an API key is configured (football), and falls
  back to the demo otherwise.

IMPORTANT: when models are enabled here it is to show the end-to-end output. On
real money, a model is enabled only after it passes the backtest harness (beats
the vig-free closing line). Every payload carries that disclaimer and the
responsible-gambling notice.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date

from .bankroll import responsible_gambling_notice
from .config import Config, ModelGate
from .data.providers import MockProvider, safe_get_fixtures
from .data.schema import Match, Sport
from .models.base import InsufficientModelData, ModelOutput
from .models.elo import EloModel
from .models.glicko import GlickoModel
from .models.poisson import DixonColesModel
from .pipeline import NoBet, Recommendation, evaluate_market

_SUPPORTED = ("football", "basketball", "tennis")

_DISCLAIMER = (
    "Models are enabled here to illustrate the end-to-end output. On real "
    "markets a model is enabled only after it beats the vig-free closing line "
    "in backtest. No outcome is ever assured."
)


def supported_sports() -> tuple[str, ...]:
    return _SUPPORTED


def _build_mock_predictor(sport: Sport, provider: MockProvider):
    """Return (market, model_key, predictor) for a sport on the mock provider."""
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
    """Enable just the one model being demonstrated, as if it had passed validation."""
    return Config(models=ModelGate(**{model_key: True}))


def _format_recommendation(r: Recommendation, match: Match) -> dict:
    row = asdict(r)
    row["match"] = f"{match.home} vs {match.away}"
    row["model_prob_pct"] = round(r.model_prob * 100, 1)
    row["vig_free_pct"] = round(r.vig_free_prob * 100, 1)
    row["edge_pts"] = round(r.edge * 100, 1)
    row["ev_pct"] = round(r.expected_value * 100, 1)
    row["band_low_pct"] = round(r.confidence_low * 100, 1)
    row["band_high_pct"] = round(r.confidence_high * 100, 1)
    row["stake_pct"] = round(r.stake_fraction * 100, 2)
    return row


def _evaluate_fixtures(fixtures, market, model_key, predictor, config, bankroll):
    """Run model -> value -> staking over fixtures; return (recs, no_bets)."""
    recommendations: list[dict] = []
    no_bets: list[dict] = []
    for match in fixtures:
        line = match.odds_for(market)
        if line is None:
            no_bets.append({
                "match": f"{match.home} vs {match.away}",
                "market": market, "selection": "—",
                "reason": "No odds available for this market — no bet.",
            })
            continue
        try:
            output: ModelOutput = predictor(match)
        except InsufficientModelData as exc:
            no_bets.append({
                "match": f"{match.home} vs {match.away}",
                "market": market, "selection": "—",
                "reason": f"Insufficient data for the model: {exc}",
            })
            continue

        for r in evaluate_market(
            match=match, model_output=output, market=market,
            decimal_odds=line.selections, bankroll=bankroll,
            model_key=model_key, config=config,
        ):
            if isinstance(r, Recommendation):
                recommendations.append(_format_recommendation(r, match))
            elif isinstance(r, NoBet):
                no_bets.append({
                    "match": f"{match.home} vs {match.away}",
                    "market": r.market, "selection": r.selection,
                    "reason": r.reason,
                })
    return recommendations, no_bets


def _base_payload(name, on, market, bankroll, source):
    return {
        "sport": name,
        "date": on.isoformat(),
        "market": market,
        "bankroll": bankroll,
        "source": source,  # "synthetic" or "api-football"
        "notice": responsible_gambling_notice(),
        "disclaimer": _DISCLAIMER,
        "recommendations": [],
        "no_bets": [],
        "error": None,
    }


def _finish(payload):
    payload["summary"] = {
        "n_value": len(payload["recommendations"]),
        "n_no_bet": len(payload["no_bets"]),
        "message": (
            f"Flagged {len(payload['recommendations'])} value bet(s). "
            "Everything else is 'no bet' — a normal and frequent outcome."
        ),
    }
    return payload


def run_demo(sport_name: str, *, seed: int = 42, on: date | None = None) -> dict:
    """Run an offline demo analysis for one sport (synthetic data)."""
    name = (sport_name or "basketball").lower()
    if name not in _SUPPORTED:
        raise ValueError(
            f"unsupported sport {name!r}; choose one of {', '.join(_SUPPORTED)}"
        )
    sport = Sport(name)
    on = on or date(2025, 3, 2)

    provider = MockProvider(seed=seed, fail_rate=0.0, thin_rate=0.25)
    market, model_key, predictor = _build_mock_predictor(sport, provider)
    config = _enabled_config(model_key)
    bankroll = config.bankroll.starting_bankroll

    payload = _base_payload(name, on, market, bankroll, "synthetic")
    fixtures, error = safe_get_fixtures(provider, sport, on)
    if error:
        payload["error"] = error  # surface failure, never fabricate fixtures
        return payload

    recs, no_bets = _evaluate_fixtures(
        fixtures, market, model_key, predictor, config, bankroll
    )
    payload["recommendations"], payload["no_bets"] = recs, no_bets
    return _finish(payload)


def run_analysis(
    sport_name: str,
    *,
    api_key: str | None = None,
    league: int | None = None,
    season: int | None = None,
    on: date | None = None,
) -> dict:
    """Analyse real fixtures when configured, else fall back to the demo.

    Live ingestion currently covers **football** via API-Football. For other
    sports, or when no key/league/season is set, this returns the synthetic
    demo (clearly labelled via ``source``) so the hosted tool always responds.
    """
    name = (sport_name or "basketball").lower()
    if name not in _SUPPORTED:
        raise ValueError(
            f"unsupported sport {name!r}; choose one of {', '.join(_SUPPORTED)}"
        )

    live_ready = bool(api_key and league and season and name == "football")
    if not live_ready:
        payload = run_demo(name, on=on)
        if name == "football" and not api_key:
            payload["note"] = (
                "Showing synthetic data. Set APIFOOTBALL_KEY, APIFOOTBALL_LEAGUE "
                "and APIFOOTBALL_SEASON to ingest real fixtures."
            )
        elif name != "football":
            payload["note"] = (
                f"Live data for {name} is not wired yet; showing synthetic data. "
                "Football supports live ingestion via API-Football."
            )
        return payload

    # --- live football path -------------------------------------------------
    from .data.apifootball import APIFootballProvider

    sport = Sport.FOOTBALL
    on = on or date.today()
    market, model_key = "1x2", "football_poisson"
    model = DixonColesModel()
    config = _enabled_config(model_key)
    bankroll = config.bankroll.starting_bankroll

    payload = _base_payload(name, on, market, bankroll, "api-football")
    provider = APIFootballProvider(api_key=api_key, league=league, season=season)
    fixtures, error = safe_get_fixtures(provider, sport, on)
    if error:
        payload["error"] = error
        return payload
    if not fixtures:
        payload["note"] = (
            f"No fixtures found for league {league}, season {season} on "
            f"{on.isoformat()}."
        )
        return _finish(payload)

    recs, no_bets = _evaluate_fixtures(
        fixtures, market, model_key, (lambda m: model.market_1x2(m)),
        config, bankroll,
    )
    payload["recommendations"], payload["no_bets"] = recs, no_bets
    return _finish(payload)
