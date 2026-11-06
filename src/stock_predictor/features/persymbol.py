"""Per-symbol features: the 26-column parity contract, plus the slower signals
from stock-analyzer's daily trend views (RSI, SMA distances, 10-day range
position, beta) that section 4 adds on top, plus any candidate inputs switched
on in the params (section 4.1).

Every daily value is as of the last completed session *before* the one being
predicted. The 10-day range position and beta are computed from daily candles
rather than read from ``v_range_context`` and ``v_symbol_beta``, which hold
only a current snapshot and so leaked the future into historical sessions. The
52-week range position is gone for the same reason: stock-analyzer keeps only a
current snapshot of the 52-week high/low, and its daily history is too short to
rebuild one point-in-time.

Every row depends only on candles at or before its own minute, so a frame built
at a later ``as_of`` agrees with an earlier one on the rows they share
(``tests/test_framestore.py`` checks this). The saved training frames rely on it.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.features.candidates import (
    CANDIDATE_COLUMNS,
    check_epoch,
    daily_candidate_values,
    opening_check_state,
)
from stock_predictor.features.parity import (
    FEATURE_NAMES,
    FeatureCandle,
    TimeOfDayVolumeProfile,
    extract_features,
)
from stock_predictor.params import Params, get_params

# Slower daily features appended to every per-candle row for a symbol.
DAILY_FEATURE_NAMES = (
    "rsi14",
    "dist_from_sma5_pct",
    "dist_from_sma20_pct",
    "dist_from_sma50_pct",
    "trend_score",
    "pct_of_10d_range",
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


def point_in_time_beta(views: AnalyzerViews, session_date: str) -> dict[str, float]:
    """Each symbol's beta to the universe, from sessions strictly before
    ``session_date``.

    The same estimator as stock-analyzer's ``v_symbol_beta``: the least-squares
    slope of a symbol's daily change % (``v_daily_change``) on the mean daily
    change of every tracked symbol that session (``v_market_breadth``). That
    view always covers every session to date, so reading it for a past session
    leaked later sessions into the value. Symbols with no defined slope are
    left out; callers default them to 1.0.
    """
    change: dict[str, dict[str, float]] = {}
    prev: dict[str, Optional[float]] = {}
    for r in views.daily_candles(before=session_date):  # ordered by symbol, session
        pc = prev.get(r.symbol)
        if pc and r.close is not None:
            change.setdefault(r.symbol, {})[r.session_date] = (r.close - pc) / pc * 100.0
        prev[r.symbol] = r.close

    by_session: dict[str, list[float]] = {}
    for per in change.values():
        for sd, v in per.items():
            by_session.setdefault(sd, []).append(v)
    market = {sd: sum(vs) / len(vs) for sd, vs in by_session.items()}

    out: dict[str, float] = {}
    for sym, per in change.items():
        n = len(per)
        xs = [market[sd] for sd in per]
        ys = list(per.values())
        sx, sy = sum(xs), sum(ys)
        den = n * sum(x * x for x in xs) - sx * sx
        if den != 0:
            out[sym] = (n * sum(x * y for x, y in zip(xs, ys)) - sx * sy) / den
    return out


def per_symbol_frame(
    views: AnalyzerViews,
    session_date: str,
    as_of: int,
    symbols: Optional[list[str]] = None,
    params: Optional[Params] = None,
) -> list[dict]:
    """One row per (symbol, minute) up to ``as_of`` with the parity features,
    the symbol's daily-trend features, and enabled candidate inputs attached."""
    params = params or get_params()
    universe = symbols or views.universe(session_date)
    sd = date.fromisoformat(session_date)

    trend = {r["symbol"]: r for r in views.symbol_trend(session_date, universe)}
    daily_pit = daily_candidate_values(views, session_date, universe, params)
    beta = point_in_time_beta(views, session_date)
    enabled = params.active_features()
    daily_cand_cols = [c for name in enabled if name != "opening_check" for c in CANDIDATE_COLUMNS[name]]
    with_oc = "opening_check" in enabled
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
        pit = daily_pit.get(sym, {})
        smas = {
            "dist_from_sma5_pct": t.get("sma5"),
            "dist_from_sma20_pct": t.get("sma20"),
            "dist_from_sma50_pct": t.get("sma50"),
        }

        daily = {
            "rsi14": t.get("rsi14") or 0.0,
            "trend_score": t.get("trend_score") if t.get("trend_score") is not None else 1.0,
            "pct_of_10d_range": pit.get("pct_of_10d_range", float("nan")),
            "beta": beta.get(sym, 1.0),
        }
        daily.update({c: pit.get(c, float("nan")) for c in daily_cand_cols})
        if with_oc:
            # Fixed once the check time has passed; unknown (NaN) on rows before it.
            oc_state, oc_vwap = opening_check_state(candles, session_date, as_of, params)
            oc_from = check_epoch(session_date, params)
        raw_by_ts = {c.ts_epoch: c for c in candles}
        for v in vectors:
            c = raw_by_ts[v["ts_epoch"]]
            row = {"symbol": sym, "session_date": session_date, "as_of": as_of, "ts_epoch": v["ts_epoch"]}
            row.update(dict(zip(FEATURE_NAMES, v["values"])))
            # Distance from the prior sessions' moving averages, at this row's
            # own close. (Using the last close before as_of here put the
            # day's later price into every earlier row of a training frame.)
            for name, ma in smas.items():
                row[name] = 0.0 if not ma else (c.close - ma) / ma * 100.0
            row.update(daily)
            if with_oc:
                known = v["ts_epoch"] >= oc_from
                row["cand_oc_range_state"] = oc_state if known else float("nan")
                row["cand_oc_above_vwap"] = oc_vwap if known else float("nan")
            # raw OHLC, underscore-prefixed so it is carried but never treated
            # as a feature (labels need it for the triple barrier).
            row["_open"], row["_high"], row["_low"], row["_close"] = c.open, c.high, c.low, c.close
            rows.append(row)
    return rows
