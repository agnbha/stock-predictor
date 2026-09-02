from __future__ import annotations

from datetime import date

from stock_predictor.clock import NSE
from stock_predictor.config import OPENING_RANGE_T1
from stock_predictor.eval.baselines import BASELINES, run_baseline
from stock_predictor.eval.backtest import backtest_baselines
from tests.conftest import requires_analyzer


@requires_analyzer
def test_every_baseline_returns_five_distinct_symbols(views, sessions):
    sd = sessions[40]
    as_of = NSE.epoch_of(date.fromisoformat(sd), OPENING_RANGE_T1)
    for name in BASELINES:
        picks = run_baseline(name, views, sd, as_of, n=5)
        assert len(picks) <= 5
        assert len({p.symbol for p in picks}) == len(picks)


@requires_analyzer
def test_random_baseline_is_deterministic_per_key(views, sessions):
    sd = sessions[40]
    as_of = NSE.epoch_of(date.fromisoformat(sd), OPENING_RANGE_T1)
    a = run_baseline("random_five", views, sd, as_of)
    b = run_baseline("random_five", views, sd, as_of)
    assert [p.symbol for p in a] == [p.symbol for p in b]


@requires_analyzer
def test_backtest_runs_and_volatility_beats_random(views, sessions):
    res = backtest_baselines(views, sessions[25:45])
    assert set(res) == set(BASELINES)
    assert all("realised_net_gain" in m for m in res.values())
    # the honest competitor should not lose to the floor on net gain
    assert res["highest_volatility"]["realised_net_gain"] >= res["random_five"]["realised_net_gain"]
