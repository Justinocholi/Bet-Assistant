"""Bet Assistant — a value-betting analysis tool, not a prediction oracle.

The public surface is intentionally small; import submodules directly for the
detailed APIs (models, odds, backtest, bankroll).
"""

__version__ = "0.1.0"

# A persistent, always-importable reminder. Surfaced by the CLI and anywhere
# results are rendered.
RESPONSIBLE_GAMBLING_NOTICE = (
    "This tool estimates probabilities and edges; it does not predict outcomes "
    "and offers no certainty. Betting risks real money you can lose. Set limits "
    "in advance and stop when you reach them. If gambling stops being fun, take "
    "a break — support is available (e.g. US: 1-800-GAMBLER)."
)
