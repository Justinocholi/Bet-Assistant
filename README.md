# Bet Assistant

A **sports betting analysis tool** for football, basketball, and tennis.

> This is an analysis aid, **not** a prediction oracle. It estimates outcome
> probabilities, compares them against bookmaker odds, and surfaces only the
> bets where the estimated probability exceeds the vig-free, odds-implied
> probability by a configurable margin (positive expected value). It never
> claims certainty, and "no bet" is a valid and frequent output.

---

## ⚠️ Responsible gambling

Betting carries real financial risk. Nothing this tool produces is a
guarantee, and no model can remove the possibility of loss. Bet only money you
can afford to lose, set limits in advance, and stop when you reach them.

The tool ships with a hard stop-loss, a persistent responsible-gambling notice,
and a self-exclusion / cool-off feature. If gambling stops being fun, take a
break. Help is available — in many countries you can call a gambling support
line (e.g. in the US: 1-800-GAMBLER).

---

## What it does

1. **Ingests** historical results, team/player stats, recent form, home/away
   splits, injuries, rest days and head-to-head data through a pluggable data
   provider interface. API failures and missing data are handled gracefully —
   the model declares *insufficient data* rather than guessing.
2. **Estimates probabilities** with well-understood baselines:
   - **Poisson / Dixon–Coles** for football scorelines and derived markets
     (1X2, Over/Under, BTTS).
   - **Elo** ratings for relative strength (basketball, football).
   - **Glicko** ratings for tennis (handles rating uncertainty directly).
   - **Logistic regression** for binary markets.
3. **Removes the vig** from bookmaker odds (normalises implied probabilities to
   sum to 1) and flags a bet as *value* only when
   `model_probability > vig_free_implied_probability + margin`.
4. **Sizes stakes** with fractional Kelly (default quarter-Kelly), capped at a
   small percentage of bankroll.
5. **Backtests** every model against held-out seasons and reports real hit
   rate, ROI and maximum drawdown. A model that cannot beat the closing line
   in backtest **ships disabled**.
6. **Checks calibration** with a reliability table — do the 60%-probability
   bets actually win ~60% of the time?

## Each flagged bet shows

- Market and selection
- Model probability vs. vig-free implied probability
- The edge / expected value, as a percentage
- A confidence band (an interval, never a single false-precise number)
- A recommended fractional-Kelly stake, capped
- Plain-language reasoning for why the model sees value

## Hard guarantees enforced in code

- Output text is **screened for forbidden certainty language**
  ("guaranteed", "sure thing", "definitely", "lock", ...). A violation raises
  an error in tests — false certainty cannot reach the user.
- Models that fail backtest validation are disabled and cannot flag bets.
- Every bet is tracked; running ROI is reported honestly, including losing
  streaks and drawdown.
- A stop-loss and self-exclusion halt staking when triggered.

## Quick start

```bash
# No third-party dependencies — pure Python standard library.
python -m bet_assistant.cli demo        # end-to-end demo on synthetic data
python -m bet_assistant.cli notice       # print the responsible-gambling notice
pytest                                    # run the test suite
```

## Layout

```
bet_assistant/
  config.py            Tunable settings (margins, Kelly fraction, caps, stop-loss)
  language.py          Forbidden-certainty-word guard + reasoning helpers
  data/                Provider interface, schema, graceful mock provider
  models/              Poisson/Dixon-Coles, Elo, Glicko, logistic regression
  odds/                Vig removal, value/EV detection, fractional Kelly
  backtest/            Backtest engine, metrics, calibration
  bankroll/            Bankroll manager, stop-loss, responsible-gambling controls
  pipeline.py          Wires data -> model -> odds -> value -> staking
  cli.py               Demo entry point
tests/                 Unit tests for the math and the guarantees
```

## Using a real data provider (API-Football)

The live adapter lives behind the same `DataProvider` interface as the offline
mock, so nothing else in the tool changes:

```python
from datetime import date
from bet_assistant.data import APIFootballProvider, safe_get_fixtures
from bet_assistant.data.schema import Sport
from bet_assistant.models.poisson import DixonColesModel

provider = APIFootballProvider(api_key="YOUR_KEY", league=39, season=2023)  # EPL
matches, error = safe_get_fixtures(provider, Sport.FOOTBALL, date(2023, 8, 1))
if error:
    print(error)            # graceful degradation — never a fabricated result
else:
    for m in matches:
        out = DixonColesModel().market_1x2(m)
        # ... feed out into the value/staking pipeline
```

Properties the adapter guarantees:

- Any transport/auth/quota failure becomes a `ProviderError`; `safe_get_fixtures`
  turns that into an empty result plus an explanatory message.
- Missing odds, form or head-to-head are recorded in `Match.quality.missing_fields`
  and never invented; thin history lowers `effective_samples`, which the value
  gate uses to emit "no bet — insufficient data".
- No third-party dependencies — it uses a small stdlib HTTP transport that is
  injectable, so the adapter is fully unit-tested offline against canned JSON.

All three sports now have live adapters behind the same interface:

- `APIFootballProvider` — fixtures + 1X2 odds + team form + head-to-head, plus
  **injuries** and **rest days**. `get_results` backfills finished fixtures for
  backtesting.
- `APIBasketballProvider` (API-Basketball) — games + moneyline odds + scoring
  form; `get_results` backfills finished games to train Elo/Glicko.
- `APITennisProvider` — configurable (`base_url`/`auth_header`) because tennis
  vendors differ; maps players + surface form + match-winner odds against a
  small documented JSON shape.

### Point-in-time backtesting (no look-ahead)

`build_pointintime_results` reconstructs each fixture's form from **only the
matches that happened before it**. Using season-aggregate stats (which include
the match being predicted) leaks the future and inflates results — this builder
avoids that, so a backtest reflects what the model could actually have known at
kickoff. Historical odds can be attached for closing-line-value measurement.

### Persistence

The bankroll ledger survives restarts:

```python
mgr.save("ledger.json")
mgr = BankrollManager.load("ledger.json", config.bankroll)
```

Bankroll, every bet, stop-loss state and any self-exclusion are persisted (atomic
write, versioned schema).

## Status of models

By default all models are **disabled until validated**. Run the backtest
harness on real historical data, confirm the model beats the vig-free closing
line, then enable it in `config.py`. This is deliberate: an unvalidated model
must never flag a live bet.
