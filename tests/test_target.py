"""Fixed targets ("touch +1% within 40 minutes"), labels over the full
session, the session in progress read live, and the target evaluation."""

from __future__ import annotations

import sqlite3
from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from stock_predictor.clock import NSE
from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.eval.target import default_points, trade_net_pct
from stock_predictor.models.registry import model_version_string
from stock_predictor.models.triple_barrier import TripleBarrierConfig, barrier_arrays
from stock_predictor.models.training import FrameSource, training_set

from tests.conftest import requires_analyzer

ONE_PCT_40 = TripleBarrierConfig(horizon_minutes=40, target_pct=1.0)


def _labels(close, high, low, cfg, ts=None, last_ts=None):
    return barrier_arrays(close, high, low, None, cfg, ts, last_ts)


def test_fixed_target_is_touching_it_within_the_horizon():
    close = [100.0] * 6
    high = [100.0, 100.5, 100.9, 101.0, 100.0, 100.0]
    low = [99.0] * 6  # far below, but there is no stop
    a = _labels(close, high, low, TripleBarrierConfig(horizon_minutes=3, target_pct=1.0))
    assert a["label"].tolist() == [1, 1, 1, 0, 0, 0]
    assert a["minutes_to_barrier"][:3].tolist() == [3, 2, 1]
    cut = _labels(close, high, low, TripleBarrierConfig(horizon_minutes=2, target_pct=1.0))
    assert cut["label"][0] == 0  # +1% came at minute 3, after the window


def test_window_is_clock_minutes_not_candles():
    # no trade at 09:17 and 09:18: the candle at 09:19 is 4 minutes after 09:15
    ts = [0, 60, 240, 300]
    close, high, low = [100.0] * 4, [100.0, 100.0, 101.5, 100.0], [100.0] * 4
    cfg = TripleBarrierConfig(horizon_minutes=3, target_pct=1.0)
    a = _labels(close, high, low, cfg, ts=ts)
    assert a["label"].tolist() == [0, 1, 0, 0]  # 09:15's window ends at 09:18
    assert a["minutes_to_barrier"][1] == 3
    assert _labels(close, high, low, cfg)["label"][0] == 1  # counting 3 candles reaches 09:19


def test_stop_first_and_a_tie_counts_as_stopped():
    cfg = TripleBarrierConfig(horizon_minutes=5, target_pct=1.0, stop_pct=1.0)
    close = [100.0] * 4
    a = _labels(close, [100, 100, 101.2, 100], [100, 98.9, 100, 100], cfg)
    assert a["label"][0] == -1 and a["minutes_to_barrier"][0] == 1
    tie = _labels(close, [100, 101.5, 100, 100], [100, 98.5, 100, 100], cfg)
    assert tie["label"][0] == -1


def test_rows_whose_window_passes_the_close_are_incomplete():
    ts = np.arange(10) * 60
    a = _labels([100.0] * 10, [100.0] * 10, [100.0] * 10,
                TripleBarrierConfig(horizon_minutes=3, target_pct=1.0), ts=ts, last_ts=9 * 60)
    assert a["complete"].tolist() == [True] * 7 + [False] * 3


def _reference(close, high, low, atr, cfg):
    """The original per-row loop, with a tie counted as the stop."""
    out = []
    n = len(close)
    for i in range(n):
        band = max(atr[i], cfg.min_barrier_pct)
        up = close[i] * (1 + cfg.upper_atr_mult * band / 100)
        dn = close[i] * (1 - cfg.lower_atr_mult * band / 100)
        end = min(i + cfg.horizon_minutes, n - 1)
        label = 0
        for j in range(i + 1, end + 1):
            if low[j] <= dn:
                label = -1
                break
            if high[j] >= up:
                label = 1
                break
        out.append((label, (close[end] - close[i]) / close[i] * 100))
    return out


def test_atr_barriers_match_the_per_row_definition():
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 400)))
    high = close * (1 + np.abs(rng.normal(0, 0.001, 400)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, 400)))
    atr = np.abs(rng.normal(0.2, 0.05, 400))
    cfg = TripleBarrierConfig(horizon_minutes=30)
    a = barrier_arrays(close, high, low, atr, cfg)
    ref = _reference(close, high, low, atr, cfg)
    assert a["label"].tolist() == [r[0] for r in ref]
    np.testing.assert_allclose(a["forward_return_pct"], [r[1] for r in ref])
    assert set(a["label"].tolist()) == {-1, 0, 1}


def test_trade_outcome_and_points():
    stop = TripleBarrierConfig(horizon_minutes=40, target_pct=1.0, stop_pct=0.5)
    got = trade_net_pct([1, -1, 0], [0.3, -2.0, 0.2], stop, 0.1)
    np.testing.assert_allclose(got, [0.9, -0.6, 0.1])
    no_stop = trade_net_pct([1, 0], [0.3, -2.0], ONE_PCT_40, 0.0)
    np.testing.assert_allclose(no_stop, [1.0, -2.0])
    pts = default_points(ONE_PCT_40)
    assert pts[0] == time(9, 30) and pts[-1] == time(14, 45)
    assert default_points(TripleBarrierConfig(horizon_minutes=60))[-1] == time(14, 15)


def test_model_versions_name_their_target():
    assert model_version_string("sklearn", "2026-11-05") == "sklearn-triple-barrier/2026-11-05"
    assert model_version_string("sklearn", "2026-11-05", ONE_PCT_40) == "sklearn-target-1pct-40m/2026-11-05"
    stop = TripleBarrierConfig(horizon_minutes=40, target_pct=1.0, stop_pct=0.5)
    assert model_version_string("sklearn", "d", stop) == "sklearn-target-1pct-stop0.5pct-40m/d"


# -- the session in progress ----------------------------------------------------


@pytest.fixture
def live_db(tmp_path):
    """Two sessions: a finished one in v_intraday_candles; today's only in
    v_intraday_merged, with its last candle still forming."""
    p = tmp_path / "sa.db"
    c = sqlite3.connect(p)
    cols = "symbol TEXT, ts_epoch INT, session_date TEXT, open REAL, high REAL, low REAL, close REAL, volume INT"
    c.execute(f"CREATE TABLE v_intraday_candles ({cols})")
    c.execute(f"CREATE TABLE v_intraday_merged ({cols}, source TEXT)")
    c.executemany("INSERT INTO v_intraday_candles VALUES (?,?,?,?,?,?,?,?)",
                  [("A", 100 + 60 * i, "2026-11-05", 1, 1, 1, 1, 10) for i in range(3)])
    c.executemany("INSERT INTO v_intraday_merged VALUES (?,?,?,?,?,?,?,?,?)",
                  [("A", 100 + 60 * i, "2026-11-05", 1, 1, 1, 1, 10, "final") for i in range(3)]
                  + [(s, 9000 + 60 * i, "2026-11-06", 1, 1, 1, 1, 10, "forming" if i == 4 else "live")
                     for s in ("A", "B") for i in range(5)])
    c.commit()
    c.close()
    return p


def test_live_views_read_the_session_in_progress(live_db):
    v = AnalyzerViews(live_db, live=True)
    v.live_session = "2026-11-06"
    assert v.universe("2026-11-06") == ["A", "B"]
    got = v.intraday_candles("2026-11-06", symbols=["B"])
    assert [c.ts_epoch for c in got] == [9000, 9060, 9120, 9180]  # forming candle left out
    assert len(v.intraday_candles("2026-11-06", as_of=9060)) == 4
    assert v.last_finished_minute("2026-11-06") == 9180
    v._live_cache[2].append(type(got[0])("A", 9240, "2026-11-06", 1, 1, 1, 1, 10))
    assert v.last_finished_minute("2026-11-06") == 9180  # one symbol ahead is not the minute
    assert v.last_finished_minute("2026-11-06", share=0.5) == 9240
    assert len(v.intraday_candles("2026-11-05")) == 3  # finished sessions as before

    plain = AnalyzerViews(live_db)
    assert plain.session_in_progress() is None
    assert plain.universe("2026-11-06") == []


# -- against the analyzer DB ----------------------------------------------------


@requires_analyzer
def test_labels_look_past_the_last_prediction_point(views, sessions, tmp_path):
    """Frames stop at 15:00; a 14:45 row's 40-minute window ends at 15:25 and
    is read from the session's candles. Rows after 14:49 are not trained on."""
    sd = sessions[-2]
    source = FrameSource(views, cfg=ONE_PCT_40)
    f = source.session(sd)
    day = date.fromisoformat(sd)
    at = NSE.epoch_of(day, time(14, 45))
    sym = f.loc[f.ts_epoch == at, "symbol"].iloc[0]
    cs = views.intraday_candles(sd, symbols=[sym])
    i = [c.ts_epoch for c in cs].index(at)
    window = [c.high for c in cs[i + 1:] if c.ts_epoch <= at + 40 * 60]
    assert max(c.ts_epoch for c in cs if c.ts_epoch <= at + 40 * 60) > NSE.epoch_of(day, time(15, 0))
    expected = int(max(window) >= cs[i].close * 1.01)
    assert int(f.loc[(f.symbol == sym) & (f.ts_epoch == at), "label_class"].iloc[0]) == expected

    complete = f[f.label_complete]
    assert complete.ts_epoch.max() <= NSE.epoch_of(day, time(14, 49))
    ts = training_set(views, [sd], source=source)
    assert ts.rows == len(complete)
    assert set(np.unique(ts.label_class)) <= {0, 1}


# -- to the close, and the best gain available ---------------------------------


def test_to_close_is_touching_the_target_before_the_close():
    from stock_predictor.models.triple_barrier import TripleBarrierConfig, barrier_arrays

    cfg = TripleBarrierConfig(target_pct=0.5, to_close=True)
    close = [100.0, 100.0, 100.4, 100.0, 99.0]
    high = [100.0, 100.2, 100.6, 100.1, 99.5]
    a = barrier_arrays(close, high, close, None, cfg, ts=[0, 60, 120, 180, 240])
    assert a["label"].tolist() == [1, 1, 0, 0, 0]  # 100.6 is +0.6% from 100, +0.2% from 100.4
    assert a["minutes_to_barrier"].tolist()[:2] == [2, 1]
    np.testing.assert_allclose(a["max_gain_pct"], [0.6, 0.6, 0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(a["forward_return_pct"], [-1.0, -1.0, (99 / 100.4 - 1) * 100, -1.0, 0.0])
    assert a["complete"].all()
    with pytest.raises(ValueError):
        TripleBarrierConfig(target_pct=0.5, stop_pct=0.5, to_close=True)
    with pytest.raises(ValueError):
        TripleBarrierConfig(to_close=True)


def test_first_reach_and_max_gain_match_a_direct_search():
    from stock_predictor.models.triple_barrier import TripleBarrierConfig, barrier_arrays, first_reach

    rng = np.random.default_rng(3)
    for _ in range(100):
        n = int(rng.integers(1, 50))
        hi, lv = rng.normal(100, 1, n), rng.normal(100.5, 1, n)
        assert first_reach(hi, lv).tolist() == [
            next((j for j in range(i + 1, n) if hi[j] >= lv[i]), n) for i in range(n)]
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, 120)))
    high = close * 1.002
    for cfg in (TripleBarrierConfig(horizon_minutes=15, target_pct=1.0), TripleBarrierConfig(horizon_minutes=15)):
        a = barrier_arrays(close, high, close * 0.998, np.full(120, 0.2), cfg)
        ref = [max([0.0] + [(high[j] / close[i] - 1) * 100 for j in range(i + 1, min(i + 16, 120))])
               for i in range(120)]
        np.testing.assert_allclose(a["max_gain_pct"], ref, atol=1e-9)


def test_to_close_versions_points_and_the_max_gain_band():
    from stock_predictor.models import ForwardModel, GBDTClassifier, GBDTQuantileRegressor

    cfg = TripleBarrierConfig(target_pct=0.5, to_close=True)
    assert model_version_string("sklearn", "d", cfg) == "sklearn-target-0.5pct-close/d"
    assert cfg.describe() == "+0.5% before the close"
    assert default_points(cfg)[-1] == time(15, 0)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    gain = np.abs(X[:, 0]) + 1.0  # the band should follow this, not the return
    ret = -gain
    m = ForwardModel(["a", "b"], GBDTClassifier(n_estimators=20), GBDTQuantileRegressor(n_estimators=40),
                     band="max_gain_pct")
    m.fit_matrix(X, (gain > 1.5).astype(int), ret, gain)
    p = m.predict(pd.DataFrame(X[:200], columns=["a", "b"]))
    assert (p.gain_p50 > 0.5).mean() > 0.95
    with pytest.raises(ValueError):
        ForwardModel(["a", "b"], band="max_gain_pct").fit_matrix(X, gain > 1, ret)


def test_the_band_is_never_out_of_order():
    from stock_predictor.models import ForwardModel, GBDTClassifier, GBDTQuantileRegressor

    rng = np.random.default_rng(5)
    X = rng.normal(size=(600, 2))
    y = np.abs(X[:, 0]) + rng.normal(0, 2, 600)  # noisy, so separately fitted quantiles cross
    m = ForwardModel(["a", "b"], GBDTClassifier(n_estimators=10), GBDTQuantileRegressor(n_estimators=60, max_depth=6))
    m.fit_matrix(X, (y > 1).astype(int), y)
    p = m.predict(pd.DataFrame(rng.normal(size=(500, 2)) * 3, columns=["a", "b"]))
    assert (p.gain_p25 <= p.gain_p50).all() and (p.gain_p50 <= p.gain_p75).all()
