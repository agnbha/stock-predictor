"""Regime features (design section 4).

Identical for every symbol on a day: market breadth, mean universe return,
dispersion, day of week, minutes since open. They exist to let the model
condition on the kind of day it is -- and they have only 66 observations, so
they are kept few and used sparingly (design section 10, "Overfitting on
regime features").
"""

from __future__ import annotations

import datetime as _dt
from statistics import pstdev

from stock_predictor.clock import NSE
from stock_predictor.data.analyzer import AnalyzerViews

REGIME_FEATURE_NAMES = (
    "regime_advancing_pct",
    "regime_mean_change_pct",
    "regime_dispersion_pct",
    "regime_minutes_since_open",
    "regime_day_of_week",
)


def regime_features(views: AnalyzerViews, session_date: str, as_of: int) -> dict:
    """Breadth/dispersion from the *prior* completed session (available at any T
    today), plus clock features for the current prediction point."""
    prior_sessions = [s for s in views.sessions(end=session_date) if s < session_date]
    breadth = views.market_breadth(prior_sessions[-1]) if prior_sessions else None

    # Cross-sectional dispersion of daily change on the prior completed session.
    disp = 0.0
    if prior_sessions:
        rows = views.symbol_trend(session_date)  # last trend row per symbol < session_date
        vals = [r["day_change_pct"] for r in rows if r.get("day_change_pct") is not None]
        if len(vals) > 1:
            disp = pstdev(vals)

    minute = max(NSE.minutes_since_open(as_of), 0)
    dow = _dt.date.fromisoformat(session_date).isoweekday()
    return {
        "regime_advancing_pct": (breadth or {}).get("advancing_pct", 50.0),
        "regime_mean_change_pct": (breadth or {}).get("mean_change_pct", 0.0),
        "regime_dispersion_pct": disp,
        "regime_minutes_since_open": float(minute),
        "regime_day_of_week": float(dow),
    }
