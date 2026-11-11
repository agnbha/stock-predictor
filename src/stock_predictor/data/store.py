"""stock-predictor's own store.

Every prediction is written here with its ``as_of`` and scored the next day
against what actually happened (design section 7, "Every prediction is
stored"). Without this, drift is invisible.

SQLite, one file, schema created on first use. Migrations are append-only
statements guarded by ``schema_version``.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

from stock_predictor.config import CONFIG

_MIGRATIONS: list[str] = [
    # 1
    """
    CREATE TABLE prediction (
        id INTEGER PRIMARY KEY,
        model_version TEXT NOT NULL,
        session_date TEXT NOT NULL,
        as_of INTEGER NOT NULL,          -- epoch the prediction was computed for (T)
        computed_at INTEGER NOT NULL,    -- epoch the row was written
        rank INTEGER NOT NULL,           -- 1..n within this (model_version, as_of)
        symbol TEXT NOT NULL,
        entry_window_start INTEGER,      -- expected entry window, epoch
        entry_window_end INTEGER,
        duration_bucket TEXT,
        gain_p25 REAL,
        gain_p50 REAL,
        gain_p75 REAL,
        score REAL,                      -- raw stage-1-derived ranking score
        realized_gain_pct REAL,          -- filled in next day
        realized_duration_minutes INTEGER,
        scored_at INTEGER,
        UNIQUE (model_version, as_of, rank)
    )
    """,
    # 2
    """
    CREATE TABLE metric_run (
        id INTEGER PRIMARY KEY,
        created_at INTEGER NOT NULL,
        kind TEXT NOT NULL,             -- 'baseline' | 'model'
        name TEXT NOT NULL,            -- baseline name or model_version
        fold TEXT,                     -- walk-forward fold id, or 'holdout'
        metric TEXT NOT NULL,
        value REAL NOT NULL,
        detail TEXT
    )
    """,
    # 3
    """
    CREATE TABLE intraday_signal (
        id INTEGER PRIMARY KEY,
        model_version TEXT NOT NULL,
        symbol TEXT NOT NULL,
        session_date TEXT NOT NULL,
        ts_epoch INTEGER NOT NULL,
        horizon_minutes INTEGER NOT NULL,
        signal TEXT NOT NULL,          -- ENTRY | EXIT | NEUTRAL
        probability REAL NOT NULL,
        reason TEXT,
        realized_return_pct REAL,
        UNIQUE (model_version, symbol, ts_epoch, horizon_minutes)
    )
    """,
    # 4: what today.sh and tomorrow.sh said, every stock, and later what happened
    """
    CREATE TABLE summary_pick (
        id INTEGER PRIMARY KEY,
        made_at INTEGER NOT NULL,          -- wall clock when the summary ran
        kind TEXT NOT NULL,                -- 'today' (rest of the session) | 'next' (the session after)
        model_version TEXT NOT NULL,
        target_pct REAL NOT NULL,
        session_date TEXT NOT NULL,        -- the session the prediction is about
        base_session TEXT NOT NULL,        -- the session whose state it used
        as_of INTEGER NOT NULL,            -- the minute of that state
        symbol TEXT NOT NULL,
        rank INTEGER NOT NULL,             -- by chance, 1 = most likely
        price REAL NOT NULL,               -- the price at as_of
        chance REAL,
        p_model REAL,
        gain_p25 REAL,
        gain_p50 REAL,
        gain_p75 REAL,
        passed INTEGER NOT NULL,           -- met the run's minimum chance
        entry_price REAL,                  -- scored: the price at as_of, or the session's open
        best_gain_pct REAL,                -- scored: the best later high against entry_price
        close_return_pct REAL,             -- scored: the close against entry_price
        hit INTEGER,                       -- scored: best_gain_pct reached target_pct
        scored_at INTEGER,
        UNIQUE (kind, model_version, session_date, as_of, symbol)
    );
    CREATE INDEX summary_pick_session ON summary_pick (session_date, kind);
    """,
]


@dataclass
class PredictionRow:
    model_version: str
    session_date: str
    as_of: int
    computed_at: int
    rank: int
    symbol: str
    entry_window_start: Optional[int] = None
    entry_window_end: Optional[int] = None
    duration_bucket: Optional[str] = None
    gain_p25: Optional[float] = None
    gain_p50: Optional[float] = None
    gain_p75: Optional[float] = None
    score: Optional[float] = None
    realized_gain_pct: Optional[float] = None
    realized_duration_minutes: Optional[int] = None
    scored_at: Optional[int] = None


class Store:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or CONFIG.store_db)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _migrate(self) -> None:
        cur = self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
        )
        row = self._conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        applied = row[0] or 0
        for i, ddl in enumerate(_MIGRATIONS[applied:], start=applied + 1):
            self._conn.executescript(ddl)
            self._conn.execute("INSERT INTO schema_version (version) VALUES (?)", (i,))
        self._conn.commit()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -- predictions -------------------------------------------------------

    def save_predictions(self, rows: list[PredictionRow]) -> None:
        cols = list(asdict(rows[0]).keys()) if rows else []
        if not rows:
            return
        placeholders = ",".join("?" * len(cols))
        sql = (
            f"INSERT OR REPLACE INTO prediction ({','.join(cols)}) VALUES ({placeholders})"
        )
        with self.tx() as c:
            c.executemany(sql, [tuple(asdict(r).values()) for r in rows])

    def latest_prediction(self, model_version: str) -> list[dict]:
        row = self._conn.execute(
            "SELECT MAX(as_of) FROM prediction WHERE model_version = ?", (model_version,)
        ).fetchone()
        if not row or row[0] is None:
            return []
        return [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM prediction WHERE model_version = ? AND as_of = ? ORDER BY rank",
                (model_version, row[0]),
            )
        ]

    def unscored_predictions(self, before_session: str) -> list[dict]:
        return [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM prediction WHERE realized_gain_pct IS NULL AND session_date < ? "
                "ORDER BY session_date, as_of, rank",
                (before_session,),
            )
        ]

    def score_prediction(
        self, pred_id: int, realized_gain_pct: float, realized_duration_minutes: Optional[int], scored_at: int
    ) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE prediction SET realized_gain_pct = ?, realized_duration_minutes = ?, "
                "scored_at = ? WHERE id = ?",
                (realized_gain_pct, realized_duration_minutes, scored_at, pred_id),
            )

    # -- metrics --------------------------------------------------------

    def record_metric(
        self,
        created_at: int,
        kind: str,
        name: str,
        fold: Optional[str],
        metric: str,
        value: float,
        detail: Optional[dict] = None,
    ) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO metric_run (created_at, kind, name, fold, metric, value, detail) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (created_at, kind, name, fold, metric, value, json.dumps(detail) if detail else None),
            )

    # -- intraday signals (model contract) --------------------------

    # -- today.sh / tomorrow.sh picks ---------------------------------------

    SUMMARY_PICK_COLUMNS = ("made_at", "kind", "model_version", "target_pct", "session_date",
                            "base_session", "as_of", "symbol", "rank", "price", "chance", "p_model",
                            "gain_p25", "gain_p50", "gain_p75", "passed")

    def save_summary_picks(self, rows: list[dict]) -> int:
        """Record a summary's predictions; a rerun for the same minute is
        ignored, so the first one made is the one scored."""
        cols = self.SUMMARY_PICK_COLUMNS
        with self.tx() as c:
            before = c.total_changes
            c.executemany(
                f"INSERT OR IGNORE INTO summary_pick ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                [tuple(r.get(k) for k in cols) for r in rows],
            )
            return c.total_changes - before

    def summary_picks(self, session_date: str, kind: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM summary_pick WHERE session_date = ?"
        args: list = [session_date]
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        return [dict(r) for r in self._conn.execute(sql + " ORDER BY as_of, rank", args)]

    def summary_pick_sessions(self) -> list[str]:
        return [r[0] for r in self._conn.execute(
            "SELECT DISTINCT session_date FROM summary_pick ORDER BY session_date")]

    def score_summary_picks(self, scores: list[dict]) -> None:
        """scores: {id, entry_price, best_gain_pct, close_return_pct, hit, scored_at}."""
        with self.tx() as c:
            c.executemany(
                "UPDATE summary_pick SET entry_price = :entry_price, best_gain_pct = :best_gain_pct, "
                "close_return_pct = :close_return_pct, hit = :hit, scored_at = :scored_at WHERE id = :id",
                scores,
            )

    def save_intraday_signals(self, rows: list[dict]) -> int:
        """Store signals for candles not stored yet; returns how many were written.

        stock-analyzer re-sends every completed candle of the session on each
        poll (every 150 s), so most rows in a request are already here. Those
        are skipped, not rewritten: a session keeps ~23k rows, and rewriting
        them on every poll cost ~1.75M row writes. Skipping also keeps the
        signal as first computed for that candle, and any ``realized_return_pct``
        filled in since, which ``REPLACE`` would have reset to NULL.
        """
        if not rows:
            return 0
        cols = ["model_version", "symbol", "session_date", "ts_epoch", "horizon_minutes",
                "signal", "probability", "reason"]
        with self.tx() as c:
            before = c.total_changes
            c.executemany(
                f"INSERT OR IGNORE INTO intraday_signal ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                [tuple(r.get(k) for k in cols) for r in rows],
            )
            return c.total_changes - before
