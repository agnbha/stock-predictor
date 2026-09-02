"""Per-symbol features: the 26-column parity contract, plus the slower signals
from stock-analyzer's daily trend views (RSI, SMA distances, 52-week range
position, beta) that section 4 adds on top.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.features.parity import (
    FEATURE_NAMES,
    FeatureCandle,
    TimeOfDayVolumeProfile,
    extract_features,
)

# Slower daily features appended to every per-candle row for a symbol.
DAILY_FEATURE_NAMES = (
    "rsi14",
    "dist_from_sma5_pct",
    "dist_from_sma20_pct",
    "dist_from_sma50_pct",
    "trend_score",
    "pct_of_10d_range",
    "pct_of_52w_range",
    "beta",
)


def _candles_for(views: AnalyzerViews, session_date: str, symbol: str, as_of: int) -> list[FeatureCandle]:
    raw = views.intraday_candles(session_date, symbols=[symbol], as_of=as_of)
    return [FeatureCandle(c.ts_epoch, c.open, c.high, c.low, c.close, c.volume) for c in raw]


def _volume_profiles(
    views: AnalyzerViews, session_date: str, symbols: list[str], lookback: int = 20
) -> dict[str, TimeOfDayVolumeProfile]:
    """Same-minute volume distribution from the prior ``lookback`` sessions, for
    every symbol at once (design section 3.2). One range query, not 20 x N."""
    prior_sessions = [s for s in views.sessions(end=session_date) if s < session_date]
    window = prior_sessions[-lookback:]
    if not window:
        return {s: TimeOfDayVolumeProfile.empty() for s in symbols}
    rows = views.intraday_candles_range(window[0], session_date, symbols)
    by_symbol: dict[str, list[FeatureCandle]] = {s: [] for s in symbols}
    for c in rows:
        if c.symbol in by_symbol:
            by_symbol[c.symbol].append(FeatureCandle(c.ts_epoch, c.open, c.high, c.low, c.close, c.volume))
    return {s: TimeOfDayVolumeProfile.from_candles(cs) for s, cs in by_symbol.items()}


def per_symbol_frame(
    views: AnalyzerViews,
    session_date: str,
    as_of: int,
    symbols: Optional[list[str]] = None,
) -> list[dict]:
    """One row per (symbol, minute) up to ``as_of`` with the parity features and
    the symbol's daily-trend features attached."""
    universe = symbols or views.universe(session_date)
    sd = date.fromisoformat(session_date)

    trend = {r["symbol"]: r for r in views.symbol_trend(session_date, universe)}
    rng = {r["symbol"]: r for r in views.range_context()}
    beta = views.symbol_beta()
    profiles = _volume_profiles(views, session_date, universe)

    rows: list[dict] = []
    for sym in universe:
        candles = _candles_for(views, session_date, sym, as_of)
        if len(candles) < 2:
            continue
        prior = views.prior_session_context(sym, session_date)
        vp = profiles.get(sym) or TimeOfDayVolumeProfile.empty()
        vectors = extract_features(
            candles,
            sd,
            prior_close=prior["close"] if prior else None,
            prior_high=prior["high"] if prior else None,
            volume_profile=vp,
        )
        t = trend.get(sym, {})
        rc = rng.get(sym, {})
        close = candles[-1].close

        def _dist(ma: Optional[float]) -> float:
            return 0.0 if not ma else (close - ma) / ma * 100.0

        daily = {
            "rsi14": t.get("rsi14") or 0.0,
            "dist_from_sma5_pct": _dist(t.get("sma5")),
            "dist_from_sma20_pct": _dist(t.get("sma20")),
            "dist_from_sma50_pct": _dist(t.get("sma50")),
            "trend_score": t.get("trend_score") if t.get("trend_score") is not None else 1.0,
            "pct_of_10d_range": rc.get("pct_of_10d_range") or 0.0,
            "pct_of_52w_range": rc.get("pct_of_52w_range") or 0.0,
            "beta": beta.get(sym, 1.0),
        }
        raw_by_ts = {c.ts_epoch: c for c in candles}
        for v in vectors:
            c = raw_by_ts[v["ts_epoch"]]
            row = {"symbol": sym, "session_date": session_date, "as_of": as_of, "ts_epoch": v["ts_epoch"]}
            row.update(dict(zip(FEATURE_NAMES, v["values"])))
            row.update(daily)
            # raw OHLC, underscore-prefixed so it is carried but never treated
            # as a feature (labels need it for the triple barrier).
            row["_open"], row["_high"], row["_low"], row["_close"] = c.open, c.high, c.low, c.close
            rows.append(row)
    return rows
