"""Assemble the stage-1 training matrix.

One row per (symbol, minute) over a set of sessions, with the full feature
frame and a triple-barrier label. Whole sessions go in or stay out -- the
walk-forward splitter works at day granularity (design section 6).
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Sequence

from stock_predictor.clock import NSE
from stock_predictor.config import LAST_INTRADAY_POINT
from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.features.pipeline import build_feature_frame
from stock_predictor.models.triple_barrier import TripleBarrierConfig, triple_barrier_labels

# Columns that are identifiers, raw inputs, or labels -- never model features.
NON_FEATURE_COLUMNS = {
    "symbol", "session_date", "as_of", "ts_epoch",
    "_open", "_high", "_low", "_close",
    "label_class", "forward_return_pct", "minutes_to_barrier",
}


def model_feature_columns(frame) -> list[str]:
    return [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]


def training_frame(
    views: AnalyzerViews,
    sessions: Sequence[str],
    cfg: Optional[TripleBarrierConfig] = None,
):
    import pandas as pd

    cfg = cfg or TripleBarrierConfig()
    parts = []
    for sd in sessions:
        # The whole session is visible for label construction; each row's
        # features were built from candles at or before that minute, and each
        # label looks forward from that minute.
        as_of = NSE.epoch_of(date.fromisoformat(sd), LAST_INTRADAY_POINT)
        frame = build_feature_frame(views, sd, as_of)
        if frame.empty:
            continue
        frame = frame.sort_values(["symbol", "ts_epoch"]).reset_index(drop=True)
        labelled = []
        for _, g in frame.groupby("symbol"):
            g = g.reset_index(drop=True)
            atr_pct = (g["atr_14"] / g["_close"].replace(0, 1) * 100.0).tolist()
            outcomes = triple_barrier_labels(
                g["_close"].tolist(), g["_high"].tolist(), g["_low"].tolist(), atr_pct, cfg
            )
            g["label_class"] = [o.label for o in outcomes]
            g["forward_return_pct"] = [o.forward_return_pct for o in outcomes]
            g["minutes_to_barrier"] = [o.minutes_to_barrier for o in outcomes]
            labelled.append(g)
        if labelled:
            parts.append(pd.concat(labelled, ignore_index=True))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
