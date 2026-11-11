"""What the frame store saves for a completed session: its feature frame,
every minute up to the last prediction point.

Kept apart from ``models/training.py`` so the saved frames' code fingerprint
(``data/framestore.py``) covers only the code that builds them. Labels are
attached on load and can change without rebuilding a frame.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from stock_predictor.clock import NSE
from stock_predictor.config import LAST_INTRADAY_POINT
from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.features.pipeline import build_feature_frame
from stock_predictor.params import Params


def training_as_of(session_date: str) -> int:
    """Saved frames run to the last prediction point. Labels look past it, to
    the session's last candle."""
    return NSE.epoch_of(date.fromisoformat(session_date), LAST_INTRADAY_POINT)


def session_features(views: AnalyzerViews, session_date: str, params: Optional[Params] = None):
    """One completed session's feature frame, every minute to the last
    prediction point."""
    return build_feature_frame(views, session_date, training_as_of(session_date), params=params)
