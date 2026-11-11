"""stock-analyzer's screener data and its own activity, for the dashboard and
for measuring screener evidence (``eval/screener.py``).

Kept out of ``data/analyzer.py``, which builds training frames: nothing here
is a model feature. The fundamentals are one snapshot per company, so as
features on past sessions they would be look-ahead. The CANSLIM runs and the
bullish score carry dates, and are joined point-in-time where they are
measured.
"""

from __future__ import annotations

import sqlite3
from typing import Optional, Sequence

from stock_predictor.data.analyzer import AnalyzerViews

# The views read here, beside the feature contract in data/analyzer.py.
SCREENER_VIEWS = (
    "v_company_fundamentals",
    "v_canslim_latest",
    "v_canslim_signal_history",
    "v_bullish_history",
    "v_trade_markers",
    "v_current_holdings",
)
# Base tables read for the same purpose: stock-analyzer publishes no view of them.
ACTIVITY_TABLES = ("monitor_heartbeat", "watchlist_pick")



class ScreenerData:
    """Read-only, over the same connection as ``views``."""

    def __init__(self, views: AnalyzerViews):
        self._conn = views._conn

    def _rows(self, sql: str, args: Sequence = ()) -> list[dict]:
        try:
            return [dict(r) for r in self._conn.execute(sql, args)]
        except sqlite3.OperationalError:  # an older stock-analyzer without that view
            return []

    def company_fundamentals(self) -> dict[str, dict]:
        """Screener.in fundamentals per symbol, as last fetched (a snapshot)."""
        return {r["symbol"]: r for r in self._rows(
            "SELECT symbol, fetched_at, results_quarter, classification, market_cap_now_cr, "
            "pe_now, pb_now, roe, roce, debt_to_equity, sales_growth_ttm, sales_growth_3y, "
            "profit_growth_ttm, profit_growth_3y, pct_from_52w_high FROM v_company_fundamentals")}

    def canslim_latest(self) -> dict[str, dict]:
        """The latest CANSLIM run's score per symbol."""
        return {r["symbol"]: r for r in self._rows(
            "SELECT symbol, as_of, run_at, market_state, letters, letters_passed, rs_rating, signal, "
            "reason, pct_from_pivot, pct_from_high, qtr_profit_growth, qtr_sales_growth, roe "
            "FROM v_canslim_latest")}

    def canslim_history(self) -> list[dict]:
        """Every CANSLIM run's score: symbol, as_of (the session it scored), signal,
        letters, rs_rating, run_at."""
        return self._rows(
            "SELECT symbol, as_of, signal, letters, rs_rating, run_at FROM v_canslim_signal_history "
            "WHERE as_of IS NOT NULL ORDER BY as_of, run_at")

    def bullish_history(self) -> list[dict]:
        """stock-analyzer's daily bullish score per symbol and session."""
        return self._rows(
            "SELECT symbol, session_date, bullish_score, rsi14, trend_score, ret5, pct_above_sma20 "
            "FROM v_bullish_history WHERE bullish_score IS NOT NULL ORDER BY session_date, symbol")

    def watchlist_picks(self) -> list[dict]:
        """stock-analyzer's daily watchlist: picks made after ``as_of_date`` for the next session."""
        return self._rows(
            "SELECT w.as_of_date, w.rank, i.symbol, w.score, w.source FROM watchlist_pick w "
            "JOIN instrument i ON i.id = w.instrument_id ORDER BY w.as_of_date, w.rank")

    def monitor_heartbeat(self) -> Optional[dict]:
        rows = self._rows("SELECT session_date, last_tick_epoch, symbols_tracked, degraded, note "
                          "FROM monitor_heartbeat")
        return rows[0] if rows else None

    def trades(self, session_date: str) -> list[dict]:
        return self._rows(
            "SELECT ts_epoch, symbol, side, product, quantity, price FROM v_trade_markers "
            "WHERE session_date = ? ORDER BY ts_epoch", (session_date,))

    def holdings(self) -> list[dict]:
        return self._rows(
            "SELECT session_date, symbol, quantity, average_price, last_price, unrealised_pct "
            "FROM v_current_holdings ORDER BY symbol")
