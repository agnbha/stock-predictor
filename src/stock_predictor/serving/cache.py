"""The in-memory answer.

A prediction for a session does not depend on who asks or when, so the model
never runs on the request path (design section 7). The scheduler writes here;
requests read. A single process with a dict is enough for one consumer.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class RankedPick:
    rank: int
    symbol: str
    entry_window_start: Optional[int]
    entry_window_end: Optional[int]
    duration_bucket: Optional[str]
    gain_p25: float
    gain_p50: float
    gain_p75: float
    score: float


@dataclass(frozen=True)
class CachedRanking:
    as_of: int                 # the moment T the prediction was computed for
    computed_at: float         # wall clock it landed in cache
    model_version: str
    session_date: str
    picks: tuple[RankedPick, ...]

    def age_seconds(self) -> float:
        return time.time() - self.computed_at


class PredictionCache:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ranking: Optional[CachedRanking] = None

    def put(self, ranking: CachedRanking) -> None:
        with self._lock:
            self._ranking = ranking

    def get(self) -> Optional[CachedRanking]:
        with self._lock:
            return self._ranking

    def top(self, n: int) -> list[RankedPick]:
        r = self.get()
        return list(r.picks[:n]) if r else []

    def for_symbol(self, symbol: str) -> Optional[RankedPick]:
        r = self.get()
        if not r:
            return None
        for p in r.picks:
            if p.symbol == symbol:
                return p
        return None
