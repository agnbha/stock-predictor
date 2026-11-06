"""Assemble the full feature frame for one prediction point.

per-symbol (parity contract + daily trend) -> cross-sectional rank/z -> regime.
Returns a pandas DataFrame; ``ts_epoch <= as_of`` throughout, so it is safe to
join to a label built for the same ``as_of``.
"""

from __future__ import annotations

from typing import Optional

from stock_predictor.clock import NSE
from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.features.crosssectional import add_cross_sectional
from stock_predictor.features.persymbol import per_symbol_frame
from stock_predictor.features.regime import regime_features
from stock_predictor.params import Params


def build_feature_frame(
    views: AnalyzerViews,
    session_date: str,
    as_of: int,
    symbols: Optional[list[str]] = None,
    params: Optional[Params] = None,
):
    import pandas as pd

    rows = per_symbol_frame(views, session_date, as_of, symbols, params)
    if not rows:
        return pd.DataFrame()
    add_cross_sectional(rows)
    regime = regime_features(views, session_date, as_of)
    for r in rows:
        r.update(regime)
        # Each row is a prediction at its own minute; giving every row the
        # as_of clock made it a constant (345) across a training frame.
        r["regime_minutes_since_open"] = float(max(NSE.minutes_since_open(r["ts_epoch"]), 0))
    df = pd.DataFrame(rows)
    return df.sort_values(["ts_epoch", "symbol"]).reset_index(drop=True)


def latest_row_per_symbol(df):
    """The single most-recent feature row per symbol -- what stage 2 scores."""
    if df.empty:
        return df
    return df.sort_values("ts_epoch").groupby("symbol", as_index=False).last()
