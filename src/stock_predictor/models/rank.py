"""One session's ranking from a trained model: features -> stage 1 -> stage 2
-> filters. Shared by ``evaluate --model``, ``ablate --model`` and serving, so
the score order and filters in the params apply the same way everywhere.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.features.pipeline import build_feature_frame
from stock_predictor.filters import apply_filters
from stock_predictor.models.aggregate import PerMinutePrediction, SymbolPrediction, aggregate_to_topn
from stock_predictor.params import Params, get_params


def rank_session(
    model,
    views: AnalyzerViews,
    session_date: str,
    as_of: int,
    n: int,
    params: Optional[Params] = None,
    frame=None,
) -> list[SymbolPrediction]:
    """The model's top ``n`` for one prediction point, after the active filters."""
    params = params or get_params()
    if frame is None:
        frame = build_feature_frame(views, session_date, as_of, params=params)
    if frame.empty:
        return []
    missing = [c for c in model.feature_names if c not in frame.columns]
    if missing:
        raise ValueError(
            f"model expects {len(missing)} column(s) the feature frame lacks, e.g. {missing[:3]}; "
            "serve and evaluate with the same candidate `feature` switches it was trained with"
        )
    pred = model.predict(frame)
    symbols = frame["symbol"].to_numpy()
    ts = frame["ts_epoch"].to_numpy()
    per_minute = [
        PerMinutePrediction(
            symbol=str(symbols[i]), ts_epoch=int(ts[i]), p_up=float(pred.p_up[i]),
            gain_p25=float(pred.gain_p25[i]), gain_p50=float(pred.gain_p50[i]),
            gain_p75=float(pred.gain_p75[i]),
        )
        for i in range(len(frame))
    ]
    latest = frame.sort_values("ts_epoch").groupby("symbol").last()
    close = latest["_close"].replace(0, np.nan)
    atr_pct = (latest["atr_14"] / close * 100.0).dropna().to_dict()
    ranked = aggregate_to_topn(
        per_minute, top_n=10**6, rank_by=params.score_order.rank_by, atr_pct=atr_pct
    )
    return apply_filters(views, session_date, as_of, ranked, n, params)
