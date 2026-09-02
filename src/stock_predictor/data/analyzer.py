"""Read-only access to stock-analyzer's published views.

The contract is the set of ``v_*`` views listed in design section 3.1, not the
tables behind them. This module opens the database in read-only mode so a bug
here can never write to it.

Every method takes an ``as_of`` epoch where leakage is possible and filters on
it in SQL; nothing downstream is trusted to remember.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Sequence

from stock_predictor.config import CONFIG

# The view contract. Anything not on this list is off-limits.
CONTRACT_VIEWS = (
    "v_intraday_candles",
    "v_daily_candles",
    "v_symbol_trend",
    "v_support_resistance",
    "v_market_breadth",
    "v_range_context",
    "v_opportunity_markers",
    "v_symbol_beta",
)


@dataclass(frozen=True)
class Candle:
    symbol: str
    ts_epoch: int
    session_date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Opportunity:
    """One window from stock-analyzer's ground-truth detector."""

    symbol: str
    session_date: str
    rank: int
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    gain_pct: float
    duration_minutes: int
    source: str  # 'final' or 'live'


class AnalyzerViews:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or CONFIG.analyzer_db)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"stock-analyzer database not found at {self.db_path}. "
                "Set STOCK_ANALYZER_DB or check out stock-analyzer beside this project."
            )
        # mode=ro: the coupling is one-way and read-only (design section 3.1).
        self._conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro&immutable=1", uri=True, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AnalyzerViews":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- calendar ---------------------------------------------------------------

    def sessions(self, start: Optional[str] = None, end: Optional[str] = None) -> list[str]:
        """Distinct trading sessions with minute data, ascending."""
        sql = "SELECT DISTINCT session_date FROM v_intraday_candles"
        clauses, args = [], []
        if start:
            clauses.append("session_date >= ?")
            args.append(start)
        if end:
            clauses.append("session_date <= ?")
            args.append(end)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY session_date"
        return [r[0] for r in self._conn.execute(sql, args)]

    def universe(self, session_date: str) -> list[str]:
        """Symbols with minute data on a given session, sorted."""
        rows = self._conn.execute(
            "SELECT DISTINCT symbol FROM v_intraday_candles WHERE session_date = ? ORDER BY symbol",
            (session_date,),
        )
        return [r[0] for r in rows]

    # -- candles --------------------------------------------------------------

    def intraday_candles(
        self,
        session_date: str,
        symbols: Optional[Sequence[str]] = None,
        as_of: Optional[int] = None,
    ) -> list[Candle]:
        """Minute candles for a session. ``as_of`` clips to ``ts_epoch <= as_of``."""
        sql = (
            "SELECT symbol, ts_epoch, session_date, open, high, low, close, volume "
            "FROM v_intraday_candles WHERE session_date = ?"
        )
        args: list = [session_date]
        if symbols:
            sql += f" AND symbol IN ({','.join('?' * len(symbols))})"
            args.extend(symbols)
        if as_of is not None:
            sql += " AND ts_epoch <= ?"
            args.append(as_of)
        sql += " ORDER BY symbol, ts_epoch"
        return [Candle(**dict(r)) for r in self._conn.execute(sql, args)]

    def intraday_candles_range(
        self,
        start_session: str,
        end_session_exclusive: str,
        symbols: Optional[Sequence[str]] = None,
    ) -> list[Candle]:
        """All minute candles in ``[start_session, end_session_exclusive)`` --
        one query, for building rolling profiles."""
        sql = (
            "SELECT symbol, ts_epoch, session_date, open, high, low, close, volume "
            "FROM v_intraday_candles WHERE session_date >= ? AND session_date < ?"
        )
        args: list = [start_session, end_session_exclusive]
        if symbols:
            sql += f" AND symbol IN ({','.join('?' * len(symbols))})"
            args.extend(symbols)
        sql += " ORDER BY symbol, ts_epoch"
        return [Candle(**dict(r)) for r in self._conn.execute(sql, args)]

    def daily_candles(
        self, symbols: Optional[Sequence[str]] = None, before: Optional[str] = None
    ) -> list[Candle]:
        """Daily candles. ``before`` clips to ``session_date < before`` (strictly prior)."""
        sql = (
            "SELECT symbol, ts_epoch, session_date, open, high, low, close, volume "
            "FROM v_daily_candles WHERE 1 = 1"
        )
        args: list = []
        if symbols:
            sql += f" AND symbol IN ({','.join('?' * len(symbols))})"
            args.extend(symbols)
        if before:
            sql += " AND session_date < ?"
            args.append(before)
        sql += " ORDER BY symbol, session_date"
        return [Candle(**dict(r)) for r in self._conn.execute(sql, args)]

    def prior_session_context(self, symbol: str, session_date: str) -> Optional[dict]:
        """Prior session's close/high/low for a symbol -- the gap and prior-day
        levels the feature contract needs."""
        row = self._conn.execute(
            "SELECT close, high, low, session_date FROM v_daily_candles "
            "WHERE symbol = ? AND session_date < ? ORDER BY session_date DESC LIMIT 1",
            (symbol, session_date),
        ).fetchone()
        return dict(row) if row else None

    # -- ground truth --------------------------------------------------------

    def opportunities(
        self,
        session_date: str,
        detector_version: Optional[str] = None,
        as_of: Optional[int] = None,
        source: str = "final",
    ) -> list[Opportunity]:
        """Detector windows for a session.

        ``as_of`` enforces the leakage rule: only windows whose **entry is
        strictly after** ``as_of`` are returned (design section 2).
        """
        dv = detector_version or CONFIG.detector_version
        sql = (
            "SELECT symbol, session_date, rank, entry_ts, exit_ts, entry_price, exit_price, "
            "gain_pct, duration_minutes, source FROM v_opportunity_markers "
            "WHERE session_date = ? AND detector_version = ?"
        )
        args: list = [session_date, dv]
        if source:
            sql += " AND source = ?"
            args.append(source)
        if as_of is not None:
            sql += " AND entry_ts > ?"
            args.append(as_of)
        sql += " ORDER BY symbol, rank"
        return [Opportunity(**dict(r)) for r in self._conn.execute(sql, args)]

    def detector_versions(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT detector_version FROM v_opportunity_markers ORDER BY detector_version"
        )
        return [r[0] for r in rows]

    # -- slower / trend features ------------------------------------------

    def symbol_trend(self, session_date: str, symbols: Optional[Sequence[str]] = None) -> list[dict]:
        """Daily SMA / RSI / trend score as of the last session strictly before
        ``session_date`` -- usable at any prediction point that day."""
        sql = (
            "SELECT t.* FROM v_symbol_trend t JOIN ("
            "  SELECT symbol, MAX(session_date) AS sd FROM v_symbol_trend "
            "  WHERE session_date < ? GROUP BY symbol"
            ") last ON last.symbol = t.symbol AND last.sd = t.session_date"
        )
        args: list = [session_date]
        if symbols:
            sql += f" WHERE t.symbol IN ({','.join('?' * len(symbols))})"
            args.extend(symbols)
        return [dict(r) for r in self._conn.execute(sql, args)]

    def support_resistance(self, session_date: str) -> list[dict]:
        """Pivot / R1..S2 / rolling highs for a session (built from prior days)."""
        return [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM v_support_resistance WHERE session_date = ?", (session_date,)
            )
        ]

    def range_context(self) -> list[dict]:
        """Latest 10-day and 52-week range position per symbol."""
        return [dict(r) for r in self._conn.execute("SELECT * FROM v_range_context")]

    def market_breadth(self, session_date: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM v_market_breadth WHERE session_date = ?", (session_date,)
        ).fetchone()
        return dict(row) if row else None

    def symbol_beta(self) -> dict[str, float]:
        return {
            r["symbol"]: r["beta"]
            for r in self._conn.execute("SELECT symbol, beta FROM v_symbol_beta")
            if r["beta"] is not None
        }

    def time_of_day_prior(self, session_date: str) -> list[dict]:
        """stock-analyzer's statistical time-of-day prior (``hot_window``), as of
        the sessions before ``session_date``. Baseline #3 in design section 5."""
        return [
            dict(r)
            for r in self._conn.execute(
                "SELECT i.symbol AS symbol, h.bucket_start_minute, h.bucket_minutes, "
                "h.lookback_days, h.hit_rate, h.hit_rate_lcb, h.mean_gain_pct, h.median_gain_pct "
                "FROM hot_window h JOIN instrument i ON i.id = h.instrument_id "
                "ORDER BY i.symbol, h.bucket_start_minute"
            )
        ]
