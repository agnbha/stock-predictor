"""The live board: what moved each probability, screener signals joined
point-in-time, stock-analyzer's requests, and the /dashboard endpoints."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_predictor.clock import NSE
from stock_predictor.eval.screener import letters_passed, reading, screener_groups
from stock_predictor.serving.live import drivers

from tests.conftest import requires_analyzer


class _Clf:
    """P rises with ret_5 and its cross-sectional rank; nothing else matters."""

    def predict_proba(self, X):
        X = np.asarray(X)
        return np.clip(0.1 + 0.05 * X[:, 0] + 0.2 * X[:, 1], 0, 1)


class _Model:
    feature_names = ["ret_5", "ret_5__xs_rank", "rsi14", "minutes_since_open"]
    _clf = _Clf()


def test_drivers_name_the_inputs_that_move_p():
    rows = pd.DataFrame({
        "ret_5": [2.0, 0.0, 0.1], "ret_5__xs_rank": [1.0, 0.0, 0.5],
        "rsi14": [70.0, 40.0, 50.0], "minutes_since_open": [300.0] * 3, "_close": [100.0] * 3,
    })
    out = drivers(_Model(), rows)
    top = out[0][0]
    assert top["feature"] == "ret_5" and top["label"] == "return, last 5 minutes"
    # ret_5 and its rank move together, against today's medians (0.1 and 0.5)
    assert top["effect_pp"] == pytest.approx((0.05 * 1.9 + 0.2 * 0.5) * 100)
    assert top["value"] == 2.0 and top["median"] == pytest.approx(0.1) and top["percent"]
    assert all(d["feature"] not in ("rsi14", "minutes_since_open") for r in out for d in r)
    assert out[1][-1]["effect_pp"] < 0  # below the median pulls P down
    assert out[2] == []  # the median stock: nothing moves it


class _Screener:
    def __init__(self, bullish, picks, canslim):
        self._b, self._p, self._c = bullish, picks, canslim

    def bullish_history(self):
        return self._b

    def watchlist_picks(self):
        return self._p

    def canslim_history(self):
        return self._c


def test_screener_groups_use_only_what_was_published_before_the_session():
    sessions = ["2026-10-27", "2026-10-28"]
    syms = [f"S{i}" for i in range(10)]
    keys = pd.DataFrame([(sd, s) for sd in sessions for s in syms], columns=["session_date", "symbol"])
    # S0 tops the bullish score on the 27th, S9 on the 28th
    bullish = [{"symbol": s, "session_date": sd, "bullish_score": float(i if sd == "2026-10-27" else -i)}
               for sd in sessions for i, s in enumerate(syms)]
    picks = [{"as_of_date": "2026-10-27", "rank": 1, "symbol": "S3"}]
    open_28 = NSE.session_open_epoch(date(2026, 10, 28))
    canslim = [{"symbol": "S1", "as_of": "2026-10-27", "rs_rating": 90, "letters": "CA-SL--", "run_at": open_28 - 3600},
               {"symbol": "S2", "as_of": "2026-10-28", "rs_rating": 95, "letters": "CANSLI-", "run_at": open_28 + 600}]
    g = screener_groups(_Screener(bullish, picks, canslim), keys).set_index(["session_date", "symbol"])

    assert np.isnan(g.loc[("2026-10-27", "S9"), "bullish_top_fifth"])  # nothing before the first session
    on_28 = g.xs("2026-10-28")
    assert set(on_28.index[on_28.bullish_top_fifth == 1]) == {"S8", "S9"}  # the 27th's scores, not the 28th's
    assert on_28.loc["S3", "watchlist_pick"] == 1 and on_28.loc["S4", "watchlist_pick"] == 0
    assert np.isnan(g.loc[("2026-10-27", "S3"), "watchlist_pick"])
    assert on_28.loc["S1", "canslim_rs_80"] == 1 and on_28.loc["S1", "canslim_4_letters"] == 1
    assert np.isnan(on_28.loc["S2", "canslim_rs_80"])  # its run finished after the open


def test_readings_and_letters():
    assert letters_passed("-A-SLI-") == 4 and letters_passed("?------") == 0 and letters_passed(None) == 0
    assert reading(1.5, 500) == "supports" and reading(0.7, 500) == "against"
    assert reading(1.05, 500) == "no measurable difference" and reading(3.0, 20) == "too little history to tell"


def test_requests_from_stock_analyzer_are_logged(store):
    from stock_predictor.serving.service import PredictorService

    svc = PredictorService(views=object(), store=store)
    rows = [{"ts_epoch": 1000 + 60 * i, "values": [0.0]} for i in range(3)]
    svc.predict_intraday("ABC", "2026-11-06", 30, ["x"], rows)
    svc.predict_intraday("ABC", "2026-11-06", 30, ["x"], rows + [{"ts_epoch": 1180, "values": [0.0]}])
    svc.predict_intraday("XYZ", "2026-11-06", 30, ["x"], rows[:1])
    s = svc.requests.summary()
    assert s["total"] == 3 and s["symbols"] == 2
    abc = next(r for r in s["by_symbol"] if r["symbol"] == "ABC")
    assert (abc["requests"], abc["rows"], abc["last_rows"], abc["last_ts_epoch"]) == (2, 7, 4, 1180)
    assert abc["last_signal"] == "NEUTRAL"


def test_dashboard_endpoints(store):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from stock_predictor.serving.app import create_app
    from stock_predictor.serving.cache import PredictionCache
    from stock_predictor.serving.service import PredictorService

    class _Board:
        def snapshot(self):
            return {"status": "ok", "candidates": [{"symbol": "ABC"}]}

    svc = PredictorService(views=object(), store=store, cache=PredictionCache())
    client = TestClient(create_app(svc, _Board()))
    page = client.get("/dashboard")
    assert page.status_code == 200 and "Likely to touch the target" in page.text
    assert client.get("/dashboard/data").json()["candidates"] == [{"symbol": "ABC"}]
    bare = TestClient(create_app(svc)).get("/dashboard/data").json()
    assert "not running" in bare["status"] and bare["requests"]["total"] == 0


@requires_analyzer
def test_board_on_a_finished_session(views, sessions):
    """Run the board on a past session as if it were today: every symbol gets
    a likelihood, its drivers, screener data and a conclusion, and the board
    holds the last minute whose window fits."""
    from stock_predictor.data.analyzer import AnalyzerViews
    from stock_predictor.models import ForwardModel, GBDTClassifier, GBDTQuantileRegressor, ModelCard
    from stock_predictor.models.training import FrameSource, training_set
    from stock_predictor.models.triple_barrier import TripleBarrierConfig
    from stock_predictor.serving.live import LiveBoard

    cfg = TripleBarrierConfig(horizon_minutes=40, target_pct=1.0)
    ts = training_set(views, [sessions[-3]], source=FrameSource(views, cfg=cfg), minute_step=5)
    model = ForwardModel(ts.feature_names, GBDTClassifier(n_estimators=20), GBDTQuantileRegressor(n_estimators=10))
    model.fit_matrix(ts.X, ts.label_class, ts.forward_return_pct)
    card = ModelCard("test", "sklearn", sessions[-3], 1, ts.feature_names, target=asdict(cfg),
                     calibration=[{"band": "0.00-1.00", "rows": 10, "predicted": 0.1, "seen": 0.1}])

    live = AnalyzerViews(views.db_path, live=True)
    live.live_session = sessions[-1]
    board = LiveBoard(live, None, model, card, "test")
    snap = board.refresh()
    json.dumps(snap)  # the endpoint returns it as JSON
    day = date.fromisoformat(sessions[-1])
    assert snap["market"]["as_of"] == NSE.session_close_epoch(day) - 60 - 40 * 60  # 14:49
    assert snap["market"]["held"]
    c = snap["candidates"]
    assert len(c) == len(views.universe(sessions[-1]))
    assert [x["rank"] for x in c] == list(range(1, len(c) + 1))
    assert all(x["conclusion"] and x["band"] for x in c)
    assert any(x["drivers"] for x in c)
    assert sum(1 for x in c if x["screener"].get("fundamentals")) > len(c) // 2
    assert snap["analyzer"]["config"]["found"] in (True, False)
