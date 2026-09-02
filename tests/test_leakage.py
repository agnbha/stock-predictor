"""Phase 1's reason for existing: prove the label is honest.

Everything downstream rests on no target window opening at or before T, and on
features never reading a candle after T.
"""

from __future__ import annotations

from datetime import date

import pytest

from stock_predictor.clock import NSE
from stock_predictor.config import OPENING_RANGE_T1, PREOPEN_T0
from stock_predictor.labels.targets import build_targets, leakage_audit, remaining_windows
from tests.conftest import requires_analyzer


@requires_analyzer
@pytest.mark.parametrize("point", [PREOPEN_T0, OPENING_RANGE_T1])
def test_no_target_opens_before_as_of(views, sessions, point):
    for sd in sessions[5:20]:
        as_of = NSE.epoch_of(date.fromisoformat(sd), point)
        targets = build_targets(views, sd, as_of, top_n=5)
        leakage_audit(targets, as_of)  # raises on any violation
        assert all(t.entry_ts > as_of for t in targets)


@requires_analyzer
def test_label_shrinks_through_the_day(views, sessions):
    """The remaining-opportunity framing: later T -> fewer windows left."""
    sd = sessions[len(sessions) // 2]
    counts = []
    for hh, mm in [(9, 30), (11, 0), (13, 0), (14, 30)]:
        as_of = NSE.epoch_of(date.fromisoformat(sd), __import__("datetime").time(hh, mm))
        counts.append(len(remaining_windows(views, sd, as_of)))
    assert counts == sorted(counts, reverse=True)


@requires_analyzer
def test_intraday_candles_respect_as_of(views, sessions):
    sd = sessions[10]
    as_of = NSE.epoch_of(date.fromisoformat(sd), OPENING_RANGE_T1)
    candles = views.intraday_candles(sd, as_of=as_of)
    assert candles
    assert max(c.ts_epoch for c in candles) <= as_of


@requires_analyzer
def test_daily_candles_are_strictly_prior(views, sessions):
    sd = sessions[20]
    rows = views.daily_candles(before=sd)
    assert rows and max(r.session_date for r in rows) < sd
