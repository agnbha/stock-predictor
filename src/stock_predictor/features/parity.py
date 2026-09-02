"""The per-candle feature contract, ported from stock-analyzer.

This is a line-for-line port of ``FeatureExtractor.java`` +
``RuleEventDetector.java`` + ``VwapCalculator.java`` +
``TimeOfDayVolumeProfile.java``. It exists so training here and serving there
compute the same 26 numbers from the same candles.

``src/test/resources/feature-parity.json`` in stock-analyzer pins the expected
vectors; ``tests/test_feature_parity.py`` asserts this port reproduces them to
1e-9. The column order is **append-only** -- reordering silently invalidates a
trained model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional, Sequence

from stock_predictor.clock import NSE, MarketClock

FEATURE_NAMES: tuple[str, ...] = (
    "minutes_since_open",
    "session_progress_sin",
    "session_progress_cos",
    "day_of_week",
    "ret_1",
    "ret_5",
    "ret_15",
    "ret_30",
    "ret_60",
    "volatility_30",
    "atr_14",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "volume_z_time_of_day",
    "dist_from_vwap_pct",
    "dist_from_open_range_high_pct",
    "dist_from_open_range_low_pct",
    "dist_from_prior_close_pct",
    "dist_from_prior_high_pct",
    "position_in_day_range",
    "gap_pct",
    "event_opening_range_breakout",
    "event_vwap_reclaim",
    "event_volume_spike",
    "event_prior_day_high_break",
)

# Event types, matching stock-analyzer's EventType enum names we need.
EV_OPENING_RANGE_BREAKOUT = "OPENING_RANGE_BREAKOUT"
EV_VWAP_RECLAIM = "VWAP_RECLAIM"
EV_VWAP_LOSS = "VWAP_LOSS"
EV_VOLUME_SPIKE = "VOLUME_SPIKE"
EV_PRIOR_DAY_HIGH_BREAK = "PRIOR_DAY_HIGH_BREAK"
EV_FIRST_HOUR_HIGH_BREAK = "FIRST_HOUR_HIGH_BREAK"
EV_GAP_AND_GO = "GAP_AND_GO"
EV_REVERSAL_AFTER_DECLINE = "REVERSAL_AFTER_DECLINE"


@dataclass(frozen=True)
class FeatureCandle:
    ts_epoch: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class TimeOfDayVolumeProfile:
    """Mean/stdev of volume per minute-of-session, from historical candles.

    Intraday volume is U-shaped, so a raw z-score mostly encodes "it is 09:20";
    comparing against the same minute on other days is what makes a spike mean
    something.
    """

    def __init__(self, mean_std_by_minute: Optional[dict[int, tuple[float, float]]] = None):
        self._stats = dict(mean_std_by_minute or {})

    @classmethod
    def empty(cls) -> "TimeOfDayVolumeProfile":
        return cls({})

    @classmethod
    def from_candles(
        cls, candles: Iterable[FeatureCandle], clock: MarketClock = NSE
    ) -> "TimeOfDayVolumeProfile":
        by_minute: dict[int, list[float]] = {}
        for c in candles:
            m = clock.minutes_since_open(c.ts_epoch)
            by_minute.setdefault(m, []).append(float(c.volume))
        stats: dict[int, tuple[float, float]] = {}
        for m, vols in by_minute.items():
            mean = sum(vols) / len(vols)
            var = sum((v - mean) ** 2 for v in vols) / len(vols)  # population, as in Java
            stats[m] = (mean, math.sqrt(var))
        return cls(stats)

    def is_empty(self) -> bool:
        return not self._stats

    def z_score(self, minutes_since_open: int, volume: float) -> float:
        s = self._stats.get(minutes_since_open)
        if s is None or s[1] <= 0:
            return 0.0
        return (volume - s[0]) / s[1]


def running_vwap(candles: Sequence[FeatureCandle]) -> list[float]:
    out: list[float] = []
    cum_val = cum_vol = 0.0
    for c in candles:
        typical = (c.high + c.low + c.close) / 3.0
        cum_val += typical * c.volume
        cum_vol += c.volume
        out.append(cum_val / cum_vol if cum_vol > 0 else c.close)
    return out


def _pct(value: float, reference: float) -> float:
    return 0.0 if reference == 0 else (value - reference) / reference * 100.0


def _return_over(candles: Sequence[FeatureCandle], index: int, lookback: int) -> float:
    frm = index - lookback
    if frm < 0:
        return 0.0
    past = candles[frm].close
    return 0.0 if past == 0 else (candles[index].close - past) / past * 100.0


def _volatility(candles: Sequence[FeatureCandle], index: int, window: int) -> float:
    frm = max(index - window + 1, 1)
    if index <= frm:
        return 0.0
    returns = []
    for i in range(frm, index + 1):
        prev = candles[i - 1].close
        if prev > 0:
            returns.append((candles[i].close - prev) / prev * 100.0)
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance)


def _atr(candles: Sequence[FeatureCandle], index: int, window: int) -> float:
    frm = max(index - window + 1, 1)
    if index < frm:
        return 0.0
    total = 0.0
    count = 0
    for i in range(frm, index + 1):
        c = candles[i]
        prev_close = candles[i - 1].close
        tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        total += tr
        count += 1
    return 0.0 if count == 0 else total / count


def detect_events(
    candles: Sequence[FeatureCandle],
    prior_close: Optional[float],
    prior_high: Optional[float],
    volume_profile: Optional[TimeOfDayVolumeProfile] = None,
    clock: MarketClock = NSE,
    opening_range_minutes: int = 15,
    volume_spike_z: float = 3.0,
    gap_pct: float = 1.0,
    reversal_candles: int = 4,
) -> set[tuple[int, str]]:
    """Port of ``RuleEventDetector.detect``. Returns the set of ``(ts_epoch,
    event_type)`` pairs; each type fires at most once per session."""
    vp = volume_profile or TimeOfDayVolumeProfile.empty()
    events: set[tuple[int, str]] = set()
    if len(candles) < 2:
        return events

    fired: set[str] = set()
    vwap = running_vwap(candles)

    opening_high = -math.inf
    opening_low = math.inf
    first_hour_high = -math.inf
    opening_range_end = -1
    for i, c in enumerate(candles):
        minute = clock.minutes_since_open(c.ts_epoch)
        if minute < opening_range_minutes:
            opening_high = max(opening_high, c.high)
            opening_low = min(opening_low, c.low)
            opening_range_end = i
        if minute < 60:
            first_hour_high = max(first_hour_high, c.high)

    first = candles[0]
    if prior_close is not None and prior_close > 0:
        gap = (first.open - prior_close) / prior_close * 100.0
        if abs(gap) >= gap_pct:
            events.add((first.ts_epoch, EV_GAP_AND_GO))
            fired.add(EV_GAP_AND_GO)

    consecutive_down = 0
    for i in range(1, len(candles)):
        c = candles[i]
        prev = candles[i - 1]
        minute = clock.minutes_since_open(c.ts_epoch)

        if (
            i > opening_range_end
            and opening_range_end >= 0
            and opening_high > -math.inf
            and c.high > opening_high
            and EV_OPENING_RANGE_BREAKOUT not in fired
        ):
            fired.add(EV_OPENING_RANGE_BREAKOUT)
            events.add((c.ts_epoch, EV_OPENING_RANGE_BREAKOUT))
        if prev.close < vwap[i - 1] and c.close > vwap[i] and EV_VWAP_RECLAIM not in fired:
            fired.add(EV_VWAP_RECLAIM)
            events.add((c.ts_epoch, EV_VWAP_RECLAIM))
        if prev.close > vwap[i - 1] and c.close < vwap[i] and EV_VWAP_LOSS not in fired:
            fired.add(EV_VWAP_LOSS)
            events.add((c.ts_epoch, EV_VWAP_LOSS))
        if vp.z_score(minute, c.volume) >= volume_spike_z and EV_VOLUME_SPIKE not in fired:
            fired.add(EV_VOLUME_SPIKE)
            events.add((c.ts_epoch, EV_VOLUME_SPIKE))
        if (
            prior_high is not None
            and c.high > prior_high
            and EV_PRIOR_DAY_HIGH_BREAK not in fired
        ):
            fired.add(EV_PRIOR_DAY_HIGH_BREAK)
            events.add((c.ts_epoch, EV_PRIOR_DAY_HIGH_BREAK))
        if (
            minute >= 60
            and first_hour_high > -math.inf
            and c.high > first_hour_high
            and EV_FIRST_HOUR_HIGH_BREAK not in fired
        ):
            fired.add(EV_FIRST_HOUR_HIGH_BREAK)
            events.add((c.ts_epoch, EV_FIRST_HOUR_HIGH_BREAK))

        if c.close < c.open:
            consecutive_down += 1
        else:
            if consecutive_down >= reversal_candles and EV_REVERSAL_AFTER_DECLINE not in fired:
                fired.add(EV_REVERSAL_AFTER_DECLINE)
                events.add((c.ts_epoch, EV_REVERSAL_AFTER_DECLINE))
            consecutive_down = 0

    return events


def extract_features(
    candles: Sequence[FeatureCandle],
    session_date: date,
    prior_close: Optional[float],
    prior_high: Optional[float],
    events: Optional[set[tuple[int, str]]] = None,
    volume_profile: Optional[TimeOfDayVolumeProfile] = None,
    clock: MarketClock = NSE,
    opening_range_minutes: int = 15,
) -> list[dict]:
    """Port of ``FeatureExtractor.extract``. One vector per candle, as a dict
    ``{"ts_epoch": int, "values": [float, ...]}`` aligned to ``FEATURE_NAMES``."""
    if not candles:
        return []
    if events is None:
        events = detect_events(
            candles, prior_close, prior_high, volume_profile, clock, opening_range_minutes
        )
    vp = volume_profile or TimeOfDayVolumeProfile.empty()
    vwap = running_vwap(candles)
    session_length = max(clock.session_length_minutes(), 1)

    opening_high = -math.inf
    opening_low = math.inf
    for c in candles:
        if clock.minutes_since_open(c.ts_epoch) < opening_range_minutes:
            opening_high = max(opening_high, c.high)
            opening_low = min(opening_low, c.low)
    if opening_high == -math.inf:
        opening_high = candles[0].high
        opening_low = candles[0].low

    gap_pct = (
        0.0
        if prior_close is None or prior_close <= 0
        else (candles[0].open - prior_close) / prior_close * 100.0
    )
    dow = float(session_date.isoweekday())

    running_high = -math.inf
    running_low = math.inf
    rows: list[dict] = []
    for i, c in enumerate(candles):
        running_high = max(running_high, c.high)
        running_low = min(running_low, c.low)
        minute = clock.minutes_since_open(c.ts_epoch)
        progress = min(max(minute / session_length, 0.0), 1.0)
        rng = max(c.high - c.low, 1e-9)
        day_range = max(running_high - running_low, 1e-9)

        values = [
            float(minute),
            math.sin(2 * math.pi * progress),
            math.cos(2 * math.pi * progress),
            dow,
            _return_over(candles, i, 1),
            _return_over(candles, i, 5),
            _return_over(candles, i, 15),
            _return_over(candles, i, 30),
            _return_over(candles, i, 60),
            _volatility(candles, i, 30),
            _atr(candles, i, 14),
            (c.close - c.open) / rng,
            (c.high - max(c.open, c.close)) / rng,
            (min(c.open, c.close) - c.low) / rng,
            vp.z_score(minute, c.volume),
            _pct(c.close, vwap[i]),
            _pct(c.close, opening_high),
            _pct(c.close, opening_low),
            0.0 if prior_close is None else _pct(c.close, prior_close),
            0.0 if prior_high is None else _pct(c.close, prior_high),
            (c.close - running_low) / day_range,
            gap_pct,
            1.0 if (c.ts_epoch, EV_OPENING_RANGE_BREAKOUT) in events else 0.0,
            1.0 if (c.ts_epoch, EV_VWAP_RECLAIM) in events else 0.0,
            1.0 if (c.ts_epoch, EV_VOLUME_SPIKE) in events else 0.0,
            1.0 if (c.ts_epoch, EV_PRIOR_DAY_HIGH_BREAK) in events else 0.0,
        ]
        rows.append({"ts_epoch": c.ts_epoch, "values": values})
    return rows
