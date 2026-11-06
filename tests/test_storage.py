"""Write volume and memory: signals written once, saved training frames, the
compact training matrix -- and the point-in-time property the saved frames
rely on (docs/ARCHITECTURE.md)."""

from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from stock_predictor.clock import NSE
from stock_predictor.data.analyzer import Candle
from stock_predictor.data.framestore import FrameStore, code_fingerprint, data_fingerprints, variant_key
from stock_predictor.features.persymbol import point_in_time_beta
from stock_predictor.features.pipeline import build_feature_frame
from stock_predictor.models.training import (
    FrameSource,
    frame_variant,
    model_feature_columns,
    training_frame,
    training_set,
)
from stock_predictor.models.triple_barrier import TripleBarrierConfig
from stock_predictor.params import Params

from tests.conftest import requires_analyzer

# -- intraday signals --------------------------------------------------------


def _poll(n_candles: int, p: float = 0.5) -> list[dict]:
    return [
        {"model_version": "m", "symbol": "ABC", "session_date": "2026-11-05",
         "ts_epoch": 1_000 + 60 * i, "horizon_minutes": 30, "signal": "NEUTRAL",
         "probability": p, "reason": None}
        for i in range(n_candles)
    ]


def test_signals_resent_on_every_poll_are_written_once(store):
    written = [store.save_intraday_signals(_poll(n, p=n / 10)) for n in (3, 5, 5, 8)]
    assert written == [3, 2, 0, 3]
    rows = store._conn.execute(
        "SELECT ts_epoch, probability FROM intraday_signal ORDER BY ts_epoch"
    ).fetchall()
    assert len(rows) == 8
    # the first prediction made for a candle is the one kept
    assert [r[1] for r in rows] == [0.3] * 3 + [0.5] * 2 + [0.8] * 3


def test_resending_keeps_realised_returns(store):
    store.save_intraday_signals(_poll(2))
    store._conn.execute("UPDATE intraday_signal SET realized_return_pct = 1.5")
    store._conn.commit()
    store.save_intraday_signals(_poll(3))
    got = store._conn.execute(
        "SELECT realized_return_pct FROM intraday_signal ORDER BY ts_epoch"
    ).fetchall()
    assert [r[0] for r in got] == [1.5, 1.5, None]


# -- frame store keys -------------------------------------------------------------


class _DigestViews:
    def __init__(self, candles: dict, daily: dict):
        self._candles, self._daily = candles, daily

    def candle_digest(self):
        return [(sd, "ABC", n, 0, 0, 0.0, 0.0, 0.0, 0.0, total)
                for sd, (n, total) in sorted(self._candles.items())]

    def daily_candles(self, symbols=None, before=None):
        return [Candle("ABC", 0, sd, c, c, c, c, 100) for sd, c in sorted(self._daily.items())]


def test_rewriting_a_session_invalidates_it_and_every_later_one():
    candles = {"2026-01-01": (375, 10.0), "2026-01-02": (375, 11.0), "2026-01-05": (375, 12.0)}
    daily = {"2026-01-01": 10.0, "2026-01-02": 11.0, "2026-01-05": 12.0}
    before = data_fingerprints(_DigestViews(candles, daily))
    candles["2026-01-02"] = (374, 11.0)  # stock-analyzer re-ingests one session
    after = data_fingerprints(_DigestViews(candles, daily))
    assert after["2026-01-01"] == before["2026-01-01"]
    assert after["2026-01-02"] != before["2026-01-02"]
    assert after["2026-01-05"] != before["2026-01-05"]
    daily["2026-01-05"] = 12.5
    assert data_fingerprints(_DigestViews(candles, daily))["2026-01-02"] == after["2026-01-02"]


def test_variant_follows_feature_switches_not_filters_or_the_target():
    base = variant_key(frame_variant(Params()))
    assert variant_key(frame_variant(Params().with_overrides({"rsi.filter": True, "rsi.min": 40.0}))) == base
    assert variant_key(frame_variant(Params().with_overrides({"rsi.feature": True}))) != base
    on = Params().with_overrides({"rsi.feature": True})
    assert variant_key(frame_variant(on.with_overrides({"rsi.period": 7}))) != variant_key(frame_variant(on))
    # labels are attached on load, so every target shares the saved frames
    assert FrameSource(None, Params(), TripleBarrierConfig(horizon_minutes=40, target_pct=1.0))._variant \
        == FrameSource(None, Params())._variant == base
    assert code_fingerprint() == code_fingerprint()


def test_store_keeps_one_file_per_session(tmp_path):
    fs = FrameStore(tmp_path)
    df = pd.DataFrame({"symbol": ["A", "B"], "ts_epoch": [1, 2], "x": [0.1, float("nan")]})
    fs.save("v1", {"features": {}}, "2026-01-02", "a" * 64, df)
    fs.save("v1", {"features": {}}, "2026-01-02", "b" * 64, df.assign(x=1.0))
    assert len(list((tmp_path / "v1").glob("*.parquet"))) == 1
    assert fs.load("v1", "2026-01-02", "a" * 64) is None
    pd.testing.assert_frame_equal(fs.load("v1", "2026-01-02", "b" * 64), df.assign(x=1.0))
    assert fs.usage()["v1"]["files"] == 1


def test_minute_step_must_be_positive():
    with pytest.raises(ValueError):
        training_set(None, [], minute_step=0)


# -- against the analyzer DB ----------------------------------------------------


@requires_analyzer
def test_frame_rows_do_not_depend_on_build_time(views, sessions):
    """Built at 15:00 or at 10:30, the rows up to 10:30 agree on every column
    but as_of. Saved frames, sliced to earlier prediction points, rely on it,
    and it is the leakage rule at row level."""
    sd = sessions[-5]
    symbols = views.universe(sd)[:8]
    params = Params().with_overrides({"opening_check.feature": True, "rsi.feature": True,
                                      "run_up.feature": True})
    day = date.fromisoformat(sd)
    full = build_feature_frame(views, sd, NSE.epoch_of(day, time(15, 0)), symbols, params)
    cut = NSE.epoch_of(day, time(10, 30))
    part = build_feature_frame(views, sd, cut, symbols, params)
    a = full[full.ts_epoch <= cut].drop(columns="as_of").reset_index(drop=True)
    b = part.drop(columns="as_of").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


@requires_analyzer
def test_point_in_time_beta_is_the_view_restricted_to_the_past(views, sessions):
    snapshot = views.symbol_beta()
    everything = point_in_time_beta(views, "9999-12-31")
    assert set(everything) == set(snapshot)
    assert all(everything[s] == pytest.approx(snapshot[s], abs=1e-9) for s in snapshot)
    early = point_in_time_beta(views, sessions[30])
    assert early and any(abs(early[s] - snapshot[s]) > 0.05 for s in early)


@pytest.fixture(scope="module")
def saved(views, sessions, tmp_path_factory):
    """One session's labelled frame, built once and saved to a scratch store."""
    params = Params().with_overrides({"opening_check.feature": True})
    store = FrameStore(tmp_path_factory.mktemp("frames"))
    sd = sessions[-3]
    built = FrameSource(views, params, store=store).session(sd)
    return params, store, sd, built


@requires_analyzer
def test_saved_frame_reads_back_unchanged(views, saved):
    params, store, sd, built = saved
    again = FrameSource(views, params, store=store)
    pd.testing.assert_frame_equal(again.session(sd), built, check_exact=True)
    assert (again.loaded, again.built) == (1, 0)
    assert not any(c in pd.read_parquet(next(store.root.rglob(f"{sd}.*.parquet"))).columns
                   for c in ("label_class", "label_complete"))  # labels are not saved


@requires_analyzer
def test_saved_frame_cut_to_a_point_equals_a_fresh_build(views, saved):
    params, store, sd, _ = saved
    source = FrameSource(views, params, store=store)
    day = date.fromisoformat(sd)
    for hh, mm in ((9, 30), (9, 45), (13, 0)):
        as_of = NSE.epoch_of(day, time(hh, mm))
        pd.testing.assert_frame_equal(
            source.at(sd, as_of), build_feature_frame(views, sd, as_of, params=params),
            check_dtype=False, check_exact=True,
        )
    assert source.built == 0


@requires_analyzer
def test_training_set_is_the_frame_as_a_matrix(views, saved):
    params, store, sd, built = saved
    source = FrameSource(views, params, store=store)
    ts = training_set(views, [sd], source=source)
    names = model_feature_columns(built)
    assert ts.feature_names == names
    complete = built[built.label_complete]  # windows that end before the close
    assert 0 < len(built) - len(complete) < len(built) * 0.01
    np.testing.assert_array_equal(ts.X, complete[names].to_numpy(dtype=float))
    np.testing.assert_array_equal(ts.label_class, complete["label_class"].to_numpy())
    assert (ts.row_session == 0).all()
    pd.testing.assert_frame_equal(training_frame(views, [sd], source=source), built)

    step = training_set(views, [sd], minute_step=5, source=source)
    assert ts.rows / 5.5 < step.rows < ts.rows / 4.5
    open_epoch = NSE.session_open_epoch(date.fromisoformat(sd))
    kept = complete[((complete.ts_epoch - open_epoch) // 60) % 5 == 0]
    np.testing.assert_array_equal(step.X, kept[names].to_numpy(dtype=float))
