"""The edge over the base rate (today.sh's flag), and +target before -stop by
the close (today.sh --stop)."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, time
from types import SimpleNamespace

import pandas as pd
import pytest

from stock_predictor.clock import NSE
from stock_predictor.data.analyzer import Candle
from stock_predictor.eval.target import by_edge, edge_record
from stock_predictor.models.registry import ModelCard, ModelRegistry, model_version_string
from stock_predictor.models.triple_barrier import TripleBarrierConfig, barrier_arrays
from stock_predictor.serving.scorecard import format_scorecard, score_pick, scorecard
from stock_predictor.serving.summary import base_rate_at, build_summary, find_model, format_summary, pick_rows

STOP = TripleBarrierConfig(target_pct=0.5, stop_pct=0.5, to_close=True)
TOUCH = TripleBarrierConfig(target_pct=0.5, to_close=True)
AT_1030 = NSE.epoch_of(date(2026, 11, 11), time(10, 30))


def test_target_before_stop_by_the_close():
    close = [100, 99.6, 99.5, 100.2, 100.6, 100.0]
    high = [100.1, 99.8, 99.7, 100.3, 100.7, 100.1]
    low = [99.9, 99.4, 99.3, 100.0, 100.4, 99.0]
    a = barrier_arrays(close, high, low, None, STOP, ts=[60 * i for i in range(6)])
    # 100: stopped at 99.4 first; 99.6 and 99.5: up at 100.3; 100.2: 100.7 is short of 100.701, then
    # stopped; 100.6: stopped; the last: nothing after it
    assert list(a["label"]) == [-1, 1, 1, -1, -1, 0]
    assert list(a["minutes_to_barrier"]) == [1, 2, 1, 2, 1, 0]
    assert a["complete"].all()
    both = barrier_arrays([100, 100], [100, 101], [100, 99], None, STOP, ts=[0, 60])
    assert both["label"][0] == -1  # one candle reaching both counts as stopped
    assert STOP.describe() == "+0.5% before -0.5%, by the close"
    with pytest.raises(ValueError):
        TripleBarrierConfig(target_pct=1, stop_pct=0.5, to_close=True, outcome="close")


def test_the_stop_model_is_named_and_found_apart(tmp_path):
    assert model_version_string("sklearn", "d", STOP) == "sklearn-target-0.5pct-stop0.5pct-close/d"
    reg = ModelRegistry(tmp_path)
    for v, cfg in (("touch", TOUCH), ("stop", STOP)):
        reg.save(v, None, ModelCard(v, "sklearn", "2026-11-13", 1, ["x"], target=asdict(cfg), band="max_gain_pct"))
    assert find_model(reg, 0.5) == "touch"
    assert find_model(reg, 0.5, stop_pct=0.5) == "stop"
    assert find_model(reg, 0.5, stop_pct=0.3) is None


def test_the_held_out_record_by_edge():
    rows = pd.DataFrame({"chance": [0.9, 0.85, 0.7, 0.62, 0.5, 0.3], "base": [0.6] * 6,
                         "hit": [True, True, False, True, False, False], "net": [0.4, 0.4, -0.5, 0.4, -0.2, -0.3]})
    b = {x["from"]: x for x in by_edge(rows)}
    assert b[20]["rows"] == 2 and b[20]["hit"] == 1.0   # +30 and +25
    assert b[10]["rows"] == 1 and b[10]["hit"] == 0.0   # +10
    assert b[0]["rows"] == 1                            # +2
    assert b[-100]["rows"] == 2                         # -10 and -30
    rec = edge_record({"by_edge": by_edge(rows)}, 10)
    assert rec["rows"] == 3 and rec["hit"] == pytest.approx(2 / 3) and rec["base"] == pytest.approx(0.6)
    assert edge_record({"by_edge": by_edge(rows)}, 40) is None and edge_record(None, 10) is None


def test_the_base_rate_follows_the_time_of_day():
    ev = {"summary": {"base_hit": 0.55},
          "by_point": [{"point": "09:30", "base_hit": 0.74}, {"point": "10:30", "base_hit": 0.59},
                       {"point": "15:00", "base_hit": 0.27}]}
    assert base_rate_at(ev, "10:30") == pytest.approx(0.59)
    assert base_rate_at(ev, "10:00") == pytest.approx((0.74 + 0.59) / 2)
    assert base_rate_at(ev, "09:20") == pytest.approx(0.74) and base_rate_at(ev, "15:20") == pytest.approx(0.27)
    assert base_rate_at(ev, "12:00", next_session=True) == 0.55
    assert base_rate_at(None, "10:30") is None


def _rows():
    return pd.DataFrame({
        "symbol": ["A", "B", "C"], "price": [100.0, 200.0, 50.0], "chance": [0.72, 0.62, 0.55],
        "p_model": [0.8, 0.7, 0.6], "p_seen": [0.72, 0.62, 0.55], "p25": [0.4, 0.3, 0.2],
        "p50": [0.9, 0.7, 0.5], "p75": [1.5, 1.2, 0.9], "vol_rank": [1, 2, 3]})


EVAL = {"model_version": "m/2026-10-13", "sessions": ["2026-10-14", "2026-11-12"], "session_count": 21,
        "cost_pct": 0.083,
        "summary": {"base_hit": 0.55, "base_net": 0.03, "model": {"hit": 0.69, "net": 0.03},
                    "intraday_volatility": {"hit": 0.69, "net": -0.02}},
        "by_point": [{"point": "09:30", "base_hit": 0.74}, {"point": "10:30", "base_hit": 0.59}],
        "by_edge": [{"from": 0, "to": 5, "rows": 900, "hit": 0.6, "base": 0.58, "net": 0.01},
                    {"from": 10, "to": 15, "rows": 120, "hit": 0.71, "base": 0.6, "net": 0.05},
                    {"from": 15, "to": 20, "rows": 30, "hit": 0.8, "base": 0.6, "net": 0.1}]}


def test_the_summary_flags_only_stocks_clearly_above_the_base_rate():
    card = SimpleNamespace(trained_through="2026-11-12", train_sessions=110)
    s = build_summary(_rows(), TOUCH, "m", card, "2026-11-13", as_of=AT_1030, finished=AT_1030,
                      latest_session="2026-11-12", evaluation=EVAL, now=AT_1030 + 60)
    assert s["edge"]["base"] == pytest.approx(0.59)
    assert [f["symbol"] for f in s["edge"]["flagged"]] == ["A"]          # +13; B is +3
    assert s["top"][1]["edge_pts"] == pytest.approx(3)
    assert s["edge"]["record"]["rows"] == 150 and s["edge"]["record"]["hit"] == pytest.approx((120 * .71 + 30 * .8) / 150)
    text = format_summary(s)
    assert "Base rate: any stock did it 59% of the time from 10:30 in the held-out month." in text
    assert "Clearly above it: A (+13)." in text
    assert "stocks +10 or more above it reached the target 73% of the time (150 checks), against 60%" in text
    assert any(ln.split()[:4] == ["#", "Stock", "Price", "Chance"] and "Edge" in ln for ln in text.splitlines())
    assert "+13 ✓" in text
    rows = pick_rows(s, made_at=1)
    assert [r["flagged"] for r in rows] == [1, 0, 0] and rows[0]["base_rate"] == pytest.approx(0.59)

    calm = build_summary(_rows().assign(chance=[0.6, 0.55, 0.5], p_seen=[0.6, 0.55, 0.5]), TOUCH, "m", card,
                         "2026-11-13", as_of=AT_1030, finished=AT_1030, latest_session="2026-11-12",
                         evaluation=EVAL, now=AT_1030 + 60)
    assert "No stock is clearly above it: the best is +1 points" in format_summary(calm)

    bare = build_summary(_rows(), TOUCH, "m", card, "2026-11-13", as_of=AT_1030, finished=AT_1030,
                         latest_session="2026-11-12", evaluation=None, now=AT_1030 + 60)
    assert bare["edge"]["base"] is None and "Edge" not in format_summary(bare)


def test_the_summary_says_what_a_stop_model_answers():
    card = SimpleNamespace(trained_through="2026-11-12", train_sessions=110)
    s = build_summary(_rows(), STOP, "m-stop", card, "2026-11-13", as_of=AT_1030, finished=AT_1030,
                      latest_session="2026-11-12", evaluation=EVAL, now=AT_1030 + 60)
    text = format_summary(s)
    assert "before today's close, before falling 0.5%" in text
    assert "0.5% above the price now before falling 0.5% (by 15:29)." in text
    assert "selling at +0.5%, at -0.5%, or else at the close" in text
    assert pick_rows(s, made_at=1)[0]["stop_pct"] == 0.5

    # a stop makes every chance smaller: a flagged stock is listed even under the 50% cut
    low = _rows().assign(chance=[0.45, 0.35, 0.3], p_seen=[0.45, 0.35, 0.3])
    ev = {**EVAL, "by_point": [{"point": "10:30", "base_hit": 0.31}]}
    s = build_summary(low, STOP, "m-stop", card, "2026-11-13", as_of=AT_1030, finished=AT_1030,
                      latest_session="2026-11-12", evaluation=ev, now=AT_1030 + 60)
    assert s["passed"] == 0 and [r["symbol"] for r in s["top"]] == ["A"]  # +14; B is +4
    assert "0 of 3 stocks have a 50% or better chance" in format_summary(s) and "(none)" not in format_summary(s)


def _candles(rows):
    day = date(2026, 11, 13)
    return [Candle("A", NSE.epoch_of(day, time(9, 15 + i)), "2026-11-13", *ohlc, 100) for i, ohlc in enumerate(rows)]


def test_a_stop_pick_hits_only_if_the_target_comes_first(store):
    cs = _candles([(100, 100.1, 99.9, 100), (100, 100.2, 99.4, 99.6), (99.6, 100.8, 99.6, 100.7)])
    pick = {"kind": "today", "as_of": cs[0].ts_epoch, "price": 100.0, "target_pct": 0.5, "id": 1}
    assert score_pick(pick, cs)["hit"] == 1                       # no stop: 100.8 reached +0.5%
    assert score_pick({**pick, "stop_pct": 0.5}, cs)["hit"] == 0  # 99.4 came first
    assert score_pick({**pick, "stop_pct": 0.7}, cs)["hit"] == 1  # 99.3 never came

    rows = [{"made_at": 1, "kind": "today", "model_version": "m-stop", "target_pct": 0.5, "session_date": "2026-11-13",
             "base_session": "2026-11-13", "as_of": cs[0].ts_epoch, "symbol": "A", "rank": 1, "price": 100.0,
             "chance": 0.72, "p_model": 0.8, "gain_p25": 0.4, "gain_p50": 0.9, "gain_p75": 1.5, "passed": 1,
             "stop_pct": 0.5, "base_rate": 0.59, "flagged": 1}]
    assert store.save_summary_picks(rows) == 1
    views = SimpleNamespace(intraday_candles=lambda s, symbols=None, as_of=None: cs, sessions=lambda: ["2026-11-13"])
    rep = scorecard(views, store, "2026-11-13", now=cs[-1].ts_epoch + 10**6)
    assert rep["runs"][0]["flagged"] == 1 and rep["runs"][0]["flagged_hits"] == 0
    text = format_scorecard(rep)
    assert "+0.5% above the price then before -0.5%" in text
    assert "Flagged as clearly above the base rate (59%): 1, 0 reached it (0%) · A ✗" in text
