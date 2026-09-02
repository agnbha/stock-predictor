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

    def save_intraday_signals(self, rows: list[dict]) -> None:
        if not rows:
            return
        cols = ["model_version", "symbol", "session_date", "ts_epoch", "horizon_minutes",
                "signal", "probability", "reason"]
        with self.tx() as c:
            c.executemany(
                f"INSERT OR REPLACE INTO intraday_signal ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                [tuple(r.get(k) for k in cols) for r in rows],
            )
