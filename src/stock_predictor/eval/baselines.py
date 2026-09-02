"""Baselines that must be beaten -- built and measured before any model, because
without them a metric is just a number (design section 5).

1. ``yesterdays_leaders``   -- the five that had the biggest windows last session
2. ``highest_volatility``   -- the five with the widest 20-day ATR (the honest one)
3. ``time_of_day_prior``    -- stock-analyzer's ``hot_window`` prior
4. ``random_five``          -- the floor

Each baseline maps ``(views, session_date, as_of)`` to an ordered list of
:class:`~stock_predictor.eval.metrics.Pick`.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

from stock_predictor.clock import NSE
from stock_predictor.config import CONFIG, duration_bucket
from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.eval.metrics import Pick

Baseline = Callable[[AnalyzerViews, str, int, int], list[Pick]]


def _prior_session(views: AnalyzerViews, session_date: str) -> Optional[str]:
    earlier = views.sessions(end=session_date)
    earlier = [s for s in earlier if s < session_date]
    return earlier[-1] if earlier else None


def yesterdays_leaders(views: AnalyzerViews, session_date: str, as_of: int, n: int) -> list[Pick]:
    prev = _prior_session(views, session_date)
    if prev is None:
        return []
    opps = views.opportunities(prev, detector_version=CONFIG.detector_version)
    best: dict[str, float] = {}
    dur: dict[str, int] = {}
    for o in opps:
        if o.gain_pct > best.get(o.symbol, float("-inf")):
            best[o.symbol] = o.gain_pct
            dur[o.symbol] = o.duration_minutes
    ranked = sorted(best, key=lambda s: best[s], reverse=True)[:n]
    return [
        Pick(symbol=s, gain_p50=best[s], duration_bucket=duration_bucket(dur[s])) for s in ranked
    ]


def _atr_20(daily: list, symbol: str) -> Optional[float]:
    rows = [c for c in daily if c.symbol == symbol]
    if len(rows) < 2:
        return None
    rows = rows[-21:]
    trs = []
    for prev, cur in zip(rows, rows[1:]):
        tr = max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close))
        trs.append(tr / prev.close * 100.0 if prev.close else 0.0)
    return sum(trs) / len(trs) if trs else None


def highest_volatility(views: AnalyzerViews, session_date: str, as_of: int, n: int) -> list[Pick]:
    universe = views.universe(session_date)
    daily = views.daily_candles(symbols=universe, before=session_date)
    scored = [(s, _atr_20(daily, s)) for s in universe]
    scored = [(s, v) for s, v in scored if v is not None]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [Pick(symbol=s, gain_p50=v) for s, v in scored[:n]]


def time_of_day_prior(views: AnalyzerViews, session_date: str, as_of: int, n: int) -> list[Pick]:
    rows = views.time_of_day_prior(session_date)
    if not rows:
        return []
    minute_now = max(NSE.minutes_since_open(as_of), 0)
    best: dict[str, float] = {}
    for r in rows:
        # only buckets that open after the current minute are still catchable
        if r["bucket_start_minute"] + r["bucket_minutes"] <= minute_now:
            continue
        expected = (r.get("mean_gain_pct") or 0.0) * (r.get("hit_rate_lcb") or r.get("hit_rate") or 0.0)
        if expected > best.get(r["symbol"], float("-inf")):
            best[r["symbol"]] = expected
    ranked = sorted(best, key=lambda s: best[s], reverse=True)[:n]
    return [Pick(symbol=s, gain_p50=best[s]) for s in ranked]


def random_five(views: AnalyzerViews, session_date: str, as_of: int, n: int) -> list[Pick]:
    universe = views.universe(session_date)
    rng = random.Random(f"{session_date}:{as_of}")
    rng.shuffle(universe)
    return [Pick(symbol=s) for s in universe[:n]]


BASELINES: dict[str, Baseline] = {
    "yesterdays_leaders": yesterdays_leaders,
    "highest_volatility": highest_volatility,
    "time_of_day_prior": time_of_day_prior,
    "random_five": random_five,
}


def run_baseline(
    name: str, views: AnalyzerViews, session_date: str, as_of: int, n: Optional[int] = None
) -> list[Pick]:
    return BASELINES[name](views, session_date, as_of, n or CONFIG.top_n)
