"""The recompute schedule (design section 7).

    scheduler -> feature build -> score -> in-memory cache -> REST
      (T0, T1, every 15 min)            -> predictions table (for later scoring)

The scheduler owns *when*; the ``recompute`` callable it is given owns *how*.
``prediction_points`` enumerates T0 (09:00), T1 (09:30) and every 15 minutes to
15:00 for a session; ``run_once`` does a single recompute; ``run_forever``
sleeps to each point and fires. Budget for one recompute: < 5 s, off the
request path.
"""

from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta
from typing import Callable, Iterable, Optional

from stock_predictor.clock import NSE
from stock_predictor.config import (
    INTRADAY_STEP_MINUTES,
    LAST_INTRADAY_POINT,
    OPENING_RANGE_T1,
    PREOPEN_T0,
)

Recompute = Callable[[str, int], None]  # (session_date, as_of_epoch) -> None


def prediction_points(session_date: str) -> list[int]:
    d = date.fromisoformat(session_date)
    pts = [NSE.epoch_of(d, PREOPEN_T0), NSE.epoch_of(d, OPENING_RANGE_T1)]
    t = datetime.combine(d, OPENING_RANGE_T1)
    last = datetime.combine(d, LAST_INTRADAY_POINT)
    t += timedelta(minutes=INTRADAY_STEP_MINUTES)
    while t <= last:
        pts.append(int(t.replace(tzinfo=NSE.tz).timestamp()))
        t += timedelta(minutes=INTRADAY_STEP_MINUTES)
    return sorted(set(pts))


class Scheduler:
    def __init__(self, recompute: Recompute):
        self._recompute = recompute
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def run_once(self, session_date: str, as_of: Optional[int] = None) -> None:
        if as_of is None:
            as_of = int(time.time())
        self._recompute(session_date, as_of)

    def run_forever(self, session_date: Optional[str] = None, poll_seconds: int = 20) -> None:
        self._thread = threading.Thread(
            target=self._loop, args=(session_date, poll_seconds), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self, session_date: Optional[str], poll_seconds: int) -> None:
        fired: set[int] = set()
        while not self._stop.is_set():
            sd = session_date or date.today().isoformat()
            now = int(time.time())
            for pt in prediction_points(sd):
                if pt <= now and pt not in fired:
                    try:
                        self._recompute(sd, pt)
                    finally:
                        fired.add(pt)
            self._stop.wait(poll_seconds)
