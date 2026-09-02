from __future__ import annotations

from datetime import date

from stock_predictor.clock import NSE
from stock_predictor.config import OPENING_RANGE_T1
from stock_predictor.labels.targets import Target, build_targets, gain_band
from tests.conftest import requires_analyzer


def test_gain_band_quantiles():
    lo, mid, hi = gain_band([0.0, 1.0, 2.0, 3.0, 4.0])
    assert (lo, mid, hi) == (1.0, 2.0, 3.0)
    assert gain_band([]) == (0.0, 0.0, 0.0)
    assert gain_band([1.5]) == (1.5, 1.5, 1.5)


@requires_analyzer
def test_targets_are_dense_ranked_and_flagged(views, sessions):
    sd = sessions[30]
    as_of = NSE.epoch_of(date.fromisoformat(sd), OPENING_RANGE_T1)
    targets = build_targets(views, sd, as_of, top_n=5)
    assert [t.universe_rank for t in targets] == list(range(1, len(targets) + 1))
    gains = [t.gain_pct for t in targets]
    assert gains == sorted(gains, reverse=True)
    assert sum(t.in_top_n for t in targets) == min(5, len(targets))
    assert all(t.duration_bucket in {"<15m", "15-60m", "60-180m", ">180m"} for t in targets)


@requires_analyzer
def test_one_target_per_symbol(views, sessions):
    sd = sessions[30]
    as_of = NSE.epoch_of(date.fromisoformat(sd), OPENING_RANGE_T1)
    targets = build_targets(views, sd, as_of, top_n=5)
    assert len({t.symbol for t in targets}) == len(targets)
