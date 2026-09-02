"""Time on the NSE session grid.

A port of ``stock-analyzer``'s ``MarketClock`` (``util/MarketClock.java``).
The exchange does not observe daylight saving, so a fixed +05:30 offset is
used rather than a named zone -- it needs no ``tzdata`` and matches the
``date(ts, 'unixepoch', '+5 hours', '+30 minutes')`` expression in
``v_intraday_candles`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)


@dataclass(frozen=True)
class MarketClock:
    """Conversions between epoch seconds and the local session grid."""

    open_time: time = SESSION_OPEN
    close_time: time = SESSION_CLOSE
    tz: timezone = IST

    def session_length_minutes(self) -> int:
        o = timedelta(hours=self.open_time.hour, minutes=self.open_time.minute)
        c = timedelta(hours=self.close_time.hour, minutes=self.close_time.minute)
        return int((c - o).total_seconds() // 60)

    def session_date_of(self, epoch_seconds: int) -> date:
        return datetime.fromtimestamp(epoch_seconds, self.tz).date()

    def time_of(self, epoch_seconds: int) -> time:
        return datetime.fromtimestamp(epoch_seconds, self.tz).timetz().replace(tzinfo=None)

    def epoch_of(self, day: date, clock_time: time) -> int:
        return int(datetime.combine(day, clock_time, self.tz).timestamp())

    def session_open_epoch(self, day: date) -> int:
        return self.epoch_of(day, self.open_time)

    def session_close_epoch(self, day: date) -> int:
        return self.epoch_of(day, self.close_time)

    def minutes_since_open(self, epoch_seconds: int) -> int:
        """Minutes since the open; negative before it."""
        day = self.session_date_of(epoch_seconds)
        return int((epoch_seconds - self.session_open_epoch(day)) // 60)

    def is_within_session(self, epoch_seconds: int) -> bool:
        t = self.time_of(epoch_seconds)
        return self.open_time <= t <= self.close_time


NSE = MarketClock()
