"""Values of the candidate screening inputs, computed point-in-time (design 4.1).

Everything daily here is built from ``v_daily_candles`` rows **strictly before**
the session being predicted, so it is known at any prediction point that day.
The opening check is built from that session's minute candles up to its check
time and is only available once the prediction point has reached it.

A value that cannot be computed -- not enough history, or a check time still in
the future -- is ``NaN``, never a guess. The trees handle ``NaN`` natively, and
a filter rule over a ``NaN`` does not apply (see :mod:`stock_predictor.filters`).

The same module computes the 10-day range position point-in-time. It replaces
the earlier read of ``v_range_context``, which holds only the *latest* row per
symbol and so leaked the future into every historical training session.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Optional, Sequence

from stock_predictor.clock import NSE
from stock_predictor.data.analyzer import AnalyzerViews, Candle
from stock_predictor.params import Params, get_params

NAN = float("nan")

# Model columns per candidate input. ``score_order`` has no column: it changes
# how the final ranking is ordered, not what a row contains.
CANDIDATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "opening_check": ("cand_oc_range_state", "cand_oc_above_vwap"),
    "run_up": ("cand_run_up_pct",),
    "rsi": ("cand_rsi",),
    "volume": ("cand_rel_volume",),
    "close_location": ("cand_close_location",),
    "momentum": ("cand_dist_from_high_pct",),
}

_RANGE_STATE_CODE = {"below": -1.0, "inside": 0.0, "above": 1.0}
_CODE_RANGE_STATE = {v: k for k, v in _RANGE_STATE_CODE.items()}


def range_state_name(code: float) -> Optional[str]:
    return None if code != code else _CODE_RANGE_STATE.get(code)  # NaN-safe


# -- daily, point-in-time --------------------------------------------------


def _rsi(closes: Sequence[float], period: int) -> float:
    """Simple-average RSI over the last ``period`` close-to-close changes --
    the definition stock-analyzer's ``v_symbol_trend.rsi14`` uses."""
    if len(closes) < period + 1:
        return NAN
    diffs = [b - a for a, b in zip(closes[-period - 1 : -1], closes[-period:])]
    gain = sum(max(d, 0.0) for d in diffs) / period
    loss = sum(max(-d, 0.0) for d in diffs) / period
    if loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + gain / loss)


def daily_values(rows: Sequence[Candle], params: Params) -> dict[str, float]:
    """Candidate values for one symbol from its daily candles, all strictly
    before the session being predicted, oldest first."""
    out: dict[str, float] = {
        "cand_run_up_pct": NAN,
        "cand_rsi": NAN,
        "cand_rel_volume": NAN,
        "cand_close_location": NAN,
        "cand_dist_from_high_pct": NAN,
        "pct_of_10d_range": NAN,
    }
    if not rows:
        return out
    closes = [r.close for r in rows]
    last = rows[-1]

    n = params.run_up.lookback_days
    if len(closes) > n and closes[-1 - n] > 0:
        out["cand_run_up_pct"] = (closes[-1] / closes[-1 - n] - 1.0) * 100.0

    out["cand_rsi"] = _rsi(closes, params.rsi.period)

    n = params.volume.lookback_days
    if len(rows) > n:
        trailing = [r.volume for r in rows[-1 - n : -1]]
        mean_vol = sum(trailing) / len(trailing)
        if mean_vol > 0:
            out["cand_rel_volume"] = last.volume / mean_vol

    if last.high > last.low:
        out["cand_close_location"] = (last.close - last.low) / (last.high - last.low)
    else:
        out["cand_close_location"] = 0.5

    n = params.momentum.lookback_days
    if len(rows) >= n:
        hi = max(r.high for r in rows[-n:])
        if hi > 0:
            out["cand_dist_from_high_pct"] = (last.close / hi - 1.0) * 100.0

    if len(rows) >= 10:
        window = rows[-10:]
        hi, lo = max(r.high for r in window), min(r.low for r in window)
        if hi > lo:
            out["pct_of_10d_range"] = (last.close - lo) / (hi - lo) * 100.0
    return out


def daily_candidate_values(
    views: AnalyzerViews, session_date: str, symbols: Sequence[str], params: Optional[Params] = None
) -> dict[str, dict[str, float]]:
    """``{symbol: {column: value}}`` from daily candles before ``session_date``."""
    params = params or get_params()
    rows = views.daily_candles(symbols=list(symbols), before=session_date)
    by_symbol: dict[str, list[Candle]] = {s: [] for s in symbols}
    for r in rows:
        if r.symbol in by_symbol:
            by_symbol[r.symbol].append(r)
    return {s: daily_values(rs, params) for s, rs in by_symbol.items()}


# -- the opening check, intraday --------------------------------------------


def check_epoch(session_date: str, params: Optional[Params] = None) -> int:
    params = params or get_params()
    return NSE.epoch_of(date.fromisoformat(session_date), params.opening_check.check_clock)


def opening_check_state(
    candles: Sequence, session_date: str, as_of: int, params: Optional[Params] = None
) -> tuple[float, float]:
    """``(range_state, above_vwap)`` at the check time for one symbol's minute
    candles (anything with ``ts_epoch``/``high``/``low``/``close``/``volume``,
    oldest first).

    ``(NaN, NaN)`` while ``as_of`` is before the check time -- the state is not
    known yet -- or if the symbol has not traded in its opening range. A symbol
    with no trade in the check minute itself uses its last price before it.
    """
    params = params or get_params()
    oc = params.opening_check
    cutoff = check_epoch(session_date, params)
    if as_of < cutoff:
        return NAN, NAN
    upto = [c for c in candles if c.ts_epoch <= cutoff]
    if not upto:
        return NAN, NAN
    or_end = NSE.session_open_epoch(date.fromisoformat(session_date)) + oc.opening_range_minutes * 60
    opening = [c for c in upto if c.ts_epoch < or_end]
    if not opening:
        return NAN, NAN
    or_high = max(c.high for c in opening)
    or_low = min(c.low for c in opening)
    last = upto[-1].close
    if last > or_high:
        state = "above"
    elif last < or_low:
        state = "below"
    else:
        state = "inside"
    pv = sum((c.high + c.low + c.close) / 3.0 * c.volume for c in upto)
    vol = sum(c.volume for c in upto)
    vwap = pv / vol if vol > 0 else last
    return _RANGE_STATE_CODE[state], 1.0 if last > vwap else 0.0


# -- everything, for one prediction point -----------------------------------------


def candidate_values(
    views: AnalyzerViews,
    session_date: str,
    as_of: int,
    symbols: Sequence[str],
    params: Optional[Params] = None,
) -> dict[str, dict[str, float]]:
    """Every candidate value for ``symbols`` as known at ``as_of``."""
    params = params or get_params()
    values = daily_candidate_values(views, session_date, symbols, params)
    oc_ready = as_of >= check_epoch(session_date, params)
    if oc_ready:
        candles = views.intraday_candles(session_date, symbols=list(symbols), as_of=as_of)
        by_symbol: dict[str, list[Candle]] = {}
        for c in candles:
            by_symbol.setdefault(c.symbol, []).append(c)
    for s in symbols:
        v = values.setdefault(s, daily_values([], params))
        state, above = opening_check_state(by_symbol.get(s, []), session_date, as_of, params) \
            if oc_ready else (NAN, NAN)
        v["cand_oc_range_state"] = state
        v["cand_oc_above_vwap"] = above
    return values


def enabled_feature_columns(params: Optional[Params] = None) -> list[str]:
    params = params or get_params()
    cols: list[str] = []
    for name in params.active_features():
        cols.extend(CANDIDATE_COLUMNS[name])
    return cols


def is_missing(x: Optional[float]) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))
