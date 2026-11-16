"""The two other outcomes: a close at least X% above the entry, and at least N
separate X% opportunities (triple_barrier.opportunity_counts)."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from stock_predictor.models import nextday
from stock_predictor.models.registry import ModelCard, ModelRegistry, model_version_string
from stock_predictor.models.triple_barrier import TripleBarrierConfig, barrier_arrays, opportunity_counts
from stock_predictor.serving.summary import find_model

# One candle a minute. Two opportunities of +0.6%: a buy at 0's low (100) sold at
# 5's high (100.7), then a buy at 6's low (99.9) sold at 11's high (100.6).
LOW = [100, 100.1, 100.1, 100.1, 100.1, 100.4, 99.9, 100, 100, 100, 100, 100.3, 100.5]
HIGH = [100.1, 100.2, 100.2, 100.2, 100.2, 100.7, 100, 100.1, 100.1, 100.1, 100.1, 100.6, 100.55]
TS = [60 * i for i in range(len(LOW))]


def test_opportunities_are_separate_round_trips_held_five_minutes():
    out = opportunity_counts(HIGH, LOW, TS, 0.6)
    assert len(out) == len(LOW) + 1 and out[-1] == 0
    assert out[0] == 2       # both
    assert out[1] == 1       # from 1 on, only the second: 100.1 needs 100.71
    assert out[7] == 0       # 100 needs 100.6 at 12 or later: 100.55
    # a sale has to come at least 5 minutes after the buy
    assert opportunity_counts([100, 101, 101, 101, 101], [100] * 5, [0, 60, 120, 180, 240], 0.6)[0] == 0
    assert opportunity_counts([100, 101, 101, 101, 101, 101], [100] * 6, [60 * i for i in range(6)], 0.6)[0] == 1
    # the next buy comes after the sale, not at it
    assert opportunity_counts([100, 100, 100, 100, 100, 101, 101, 101, 101, 101, 102],
                              [100] * 5 + [100.4] + [101] * 5, [60 * i for i in range(11)], 0.6)[0] == 1


def test_a_steady_rise_counts_as_two_when_it_could_be_traded_twice():
    # 100 -> 100.65 -> 101.3 in two legs: hindsight could take +0.6% twice
    low = [100, 100.1, 100.2, 100.3, 100.4, 100.6, 100.65, 100.7, 100.8, 100.9, 101.0, 101.2]
    high = [x + 0.05 for x in low]
    high[5], high[11] = 100.65, 101.3
    assert opportunity_counts(high, low, [60 * i for i in range(12)], 0.6)[0] == 2


def test_close_and_opportunity_labels_per_row():
    close = [100.0, 100.5, 101.2]
    a = barrier_arrays(close, close, close, None, TripleBarrierConfig(target_pct=1, to_close=True, outcome="close"),
                       ts=[0, 60, 120])
    assert list(a["label"]) == [1, 0, 0]  # 101.2 is +1.2% on 100, +0.7% on 100.5
    assert a["forward_return_pct"][0] == pytest.approx(1.2)

    cfg = TripleBarrierConfig(target_pct=0.6, to_close=True, outcome="opportunities", count=2)
    closes = [(h + lo) / 2 for h, lo in zip(HIGH, LOW)]
    b = barrier_arrays(closes, HIGH, LOW, None, cfg, ts=TS)
    counts = opportunity_counts(HIGH, LOW, TS, 0.6)[1:]  # buys after each row's own candle
    assert list(b["opportunities"]) == list(counts)
    assert list(b["label"]) == list((counts >= 2).astype(int))
    assert b["complete"].all()


def test_outcome_settings_are_checked_and_described():
    with pytest.raises(ValueError):
        TripleBarrierConfig(target_pct=1, outcome="close")  # needs a window to the close
    with pytest.raises(ValueError):
        TripleBarrierConfig(target_pct=1, to_close=True, count=2)  # count is for opportunities
    with pytest.raises(ValueError):
        TripleBarrierConfig(target_pct=1, to_close=True, outcome="best")
    assert TripleBarrierConfig(target_pct=1, to_close=True, outcome="close").describe() == \
        "a close +1% or more above the entry"
    opps = TripleBarrierConfig(target_pct=0.6, to_close=True, outcome="opportunities", count=2)
    assert opps.describe() == "2+ separate +0.6% opportunities before the close"


def test_models_for_other_outcomes_are_named_apart_and_never_taken_for_today_sh(tmp_path):
    close = TripleBarrierConfig(target_pct=1, to_close=True, outcome="close")
    opps = TripleBarrierConfig(target_pct=0.6, to_close=True, outcome="opportunities", count=2)
    touch = TripleBarrierConfig(target_pct=1, to_close=True)
    assert model_version_string("sklearn", "d", close) == "sklearn-target-1pct-at-close/d"
    assert model_version_string("sklearn", "d", opps) == "sklearn-target-0.6pct-close-2x/d"
    assert model_version_string("sklearn", "d", touch) == "sklearn-target-1pct-close/d"
    assert model_version_string("sklearn", "d", close, next_session=True) == "sklearn-nextday-1pct-at-close/d"
    assert model_version_string("sklearn", "d", opps, next_session=True) == "sklearn-nextday-0.6pct-2x/d"

    reg = ModelRegistry(tmp_path)
    for v, cfg in (("close", close), ("opps", opps)):
        reg.save(v, None, ModelCard(v, "sklearn", "2026-11-12", 1, ["x"], target=asdict(cfg), band="max_gain_pct"))
    assert find_model(reg, 1.0) is None and find_model(reg, 0.6) is None  # today.sh wants "touch"
    assert reg.latest_with_target() is None                              # nor does the dashboard
    reg.save("touch", None, ModelCard("touch", "sklearn", "2026-11-11", 1, ["x"], target=asdict(touch),
                                      band="max_gain_pct"))
    assert find_model(reg, 1.0) == "touch" and reg.latest_with_target() == "touch"


def test_a_card_saved_before_outcomes_existed_still_matches():
    old = {"horizon_minutes": 30, "upper_atr_mult": 1.0, "lower_atr_mult": 1.0, "min_barrier_pct": 0.15,
           "target_pct": 0.5, "stop_pct": None, "to_close": True}
    card = ModelCard("m", "sklearn", "d", 1, ["x"], target=old, band="max_gain_pct")
    assert card.target == asdict(TripleBarrierConfig(target_pct=0.5, to_close=True))
    assert card.target["outcome"] == "touch" and card.target["count"] == 1
    new = ModelCard("n", "sklearn", "d", 1, ["x"], target=asdict(TripleBarrierConfig(target_pct=0.5, to_close=True)),
                    band="max_gain_pct")
    assert card.recipe() == new.recipe()  # a twin evaluated before still describes a live model after


def test_next_session_outcomes_and_that_they_are_never_features():
    rows = pd.DataFrame({"symbol": ["A", "B", "C"], "x": [1.0, 2.0, 3.0], "session_close": [101.5, 99.0, 50.0],
                         "prev_session_close": [100.0, 100.0, np.nan], "opportunities": [3, 1, 2],
                         "label_class": [0, 1, 1], "max_gain_pct": [1.0, 0.2, 0.9], "close_return_pct": [0.1] * 3})
    close = nextday.apply_outcome(rows.copy(), 1.0, "close")
    assert list(close["symbol"]) == ["A", "B"]  # C has no close the session before
    assert list(close["label_class"]) == [1, 0] and close["close_vs_prev_pct"].iloc[0] == pytest.approx(1.5)
    opps = nextday.apply_outcome(rows.copy(), 0.6, "opportunities", 2)
    assert list(opps["label_class"]) == [1, 0, 1]
    assert list(nextday.apply_outcome(rows.copy(), 0.5)["label_class"]) == [0, 1, 1]  # touch: unchanged
    assert nextday.feature_names(close) == ["x"]
    assert nextday.band_of("close") == ("forward_return_pct", "close_vs_prev_pct")
    assert nextday.band_of("opportunities") == ("max_gain_pct", "max_gain_pct")
