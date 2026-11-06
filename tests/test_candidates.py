"""Candidate screening inputs (design 4.1): params, point-in-time values,
filter rules, the adverse-rate metric, and the score-order switch."""

from __future__ import annotations

import json
import math
from datetime import date, time, timedelta

import pytest

from stock_predictor.clock import NSE
from stock_predictor.data.analyzer import Candle
from stock_predictor.eval.metrics import Pick, adverse_rate, evaluate
from stock_predictor.features.candidates import (
    candidate_values,
    daily_values,
    opening_check_state,
    range_state_name,
)
from stock_predictor.filters import apply_filters, evaluate_rule, failed_rules
from stock_predictor.models.aggregate import PerMinutePrediction, aggregate_to_topn
from stock_predictor.params import Params, load_params, parse_set

SESSION = "2026-10-01"
DAY = date.fromisoformat(SESSION)


def _daily(symbol: str, closes, start=date(2026, 5, 1), high_pad=1.0, low_pad=1.0, volumes=None):
    rows = []
    for i, c in enumerate(closes):
        d = (start + timedelta(days=i)).isoformat()
        v = volumes[i] if volumes else 1000.0
        rows.append(Candle(symbol, 0, d, c, c + high_pad, c - low_pad, c, v))
    return rows


def _minute(symbol: str, hhmm: str, o, h, l, c, v=100.0) -> Candle:
    hh, mm = map(int, hhmm.split(":"))
    return Candle(symbol, NSE.epoch_of(DAY, time(hh, mm)), SESSION, o, h, l, c, v)


class FakeViews:
    """Just enough of AnalyzerViews for candidate_values / apply_filters."""

    def __init__(self, daily, intraday):
        self._daily = daily
        self._intraday = intraday
        self.daily_calls = []

    def daily_candles(self, symbols=None, before=None):
        self.daily_calls.append(before)
        return [r for r in self._daily
                if (not symbols or r.symbol in symbols) and (before is None or r.session_date < before)]

    def intraday_candles(self, session_date, symbols=None, as_of=None):
        return [c for c in self._intraday
                if c.session_date == session_date and (not symbols or c.symbol in symbols)
                and (as_of is None or c.ts_epoch <= as_of)]


# -- params -----------------------------------------------------------------


def test_defaults_switch_everything_off():
    p = Params()
    assert p.active_filters() == []
    assert p.active_features() == []
    assert p.score_order.rank_by == "score"
    assert set(p.candidates()) == {"opening_check", "run_up", "rsi", "volume",
                                   "close_location", "momentum"}


def test_overrides_coerce_types_and_reject_unknowns():
    p = Params().with_overrides(parse_set([
        "rsi.filter=true", "rsi.max=65", "run_up.lookback_days=10",
        "opening_check.allowed_range_states=above,inside",
    ]))
    assert p.rsi.filter is True and p.rsi.max == 65.0
    assert p.run_up.lookback_days == 10
    assert p.opening_check.allowed_range_states == ("above", "inside")
    assert p.active_filters() == ["rsi"]
    with pytest.raises(ValueError):
        Params().with_overrides({"rsi.maxx": "1"})
    with pytest.raises(ValueError):
        Params().with_overrides({"nope.x": "1"})
    with pytest.raises(ValueError):
        Params().with_overrides({"score_order.rank_by": "magic"})
    with pytest.raises(ValueError):
        Params().with_overrides({"opening_check.allowed_range_states": "sideways"})


def test_params_file_round_trip(tmp_path):
    toml = tmp_path / "p.toml"
    toml.write_text('[volume]\nfilter = true\nmin_relative_volume = 1.5\n'
                    '[score_order]\nrank_by = "risk_adjusted"\n')
    p = load_params(toml)
    assert p.volume.filter and p.volume.min_relative_volume == 1.5
    assert p.score_order.rank_by == "risk_adjusted"
    js = tmp_path / "p.json"
    js.write_text(json.dumps(p.to_dict()))
    assert load_params(js) == p


def test_example_params_file_matches_defaults():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "params.example.toml"
    assert load_params(example) == Params()


# -- daily values -------------------------------------------------------------


def test_daily_values_formulas():
    closes = [100.0] * 15 + [100, 102, 104, 106, 108, 110]
    p = Params()
    v = daily_values(_daily("A", closes), p)
    assert v["cand_run_up_pct"] == pytest.approx((110 / 100 - 1) * 100)  # 5 sessions back
    assert v["cand_rsi"] == pytest.approx(100.0)  # only gains in the last 14 changes
    assert v["cand_close_location"] == pytest.approx(0.5)  # close mid-range in the fixture
    assert math.isnan(v["cand_dist_from_high_pct"])  # 21 rows < 120-day lookback
    assert v["cand_rel_volume"] == pytest.approx(1.0)  # 20 prior sessions + the last one
    v2 = daily_values(_daily("A", closes), p.with_overrides({"momentum.lookback_days": "10"}))
    assert v2["cand_dist_from_high_pct"] == pytest.approx((110 / 111 - 1) * 100)


def test_relative_volume_and_close_location():
    vols = [1000.0] * 20 + [3000.0]
    rows = _daily("A", [100.0] * 21, volumes=vols)
    rows[-1] = Candle("A", 0, rows[-1].session_date, 100, 110, 100, 108, 3000.0)
    v = daily_values(rows, Params())
    assert v["cand_rel_volume"] == pytest.approx(3.0)
    assert v["cand_close_location"] == pytest.approx(0.8)


def test_daily_values_never_see_the_session_itself():
    """Point-in-time: the session's own daily row (and anything later) is excluded."""
    hist = _daily("A", [100.0] * 30, start=date(2026, 9, 1))  # runs through 2026-09-30
    poison = [Candle("A", 0, SESSION, 999, 999, 999, 999, 1e9),
              Candle("A", 0, "2026-10-02", 999, 999, 999, 999, 1e9)]
    views = FakeViews(hist + poison, [])
    v = candidate_values(views, SESSION, NSE.epoch_of(DAY, time(9, 30)), ["A"], Params())["A"]
    assert views.daily_calls == [SESSION]
    assert v["cand_run_up_pct"] == pytest.approx(0.0)
    assert v["cand_rel_volume"] == pytest.approx(1.0)


# -- opening check ---------------------------------------------------------------


def _opening_session(symbol: str, last_close: float):
    """Opening range 100-101 (09:15-09:29), then drift to ``last_close`` by 09:45."""
    cs = [_minute(symbol, f"09:{m:02d}", 100.5, 101.0, 100.0, 100.5) for m in range(15, 30)]
    cs += [_minute(symbol, f"09:{m:02d}", last_close, last_close, last_close, last_close)
           for m in range(30, 46)]
    return cs


def test_opening_check_states():
    p = Params()
    at = NSE.epoch_of(DAY, time(9, 45))
    s, vw = opening_check_state(_opening_session("A", 102.0), SESSION, at, p)
    assert range_state_name(s) == "above" and vw == 1.0
    s, vw = opening_check_state(_opening_session("A", 99.0), SESSION, at, p)
    assert range_state_name(s) == "below" and vw == 0.0
    s, _ = opening_check_state(_opening_session("A", 100.5), SESSION, at, p)
    assert range_state_name(s) == "inside"


def test_opening_check_unknown_before_check_time():
    p = Params()
    before = NSE.epoch_of(DAY, time(9, 44))
    s, vw = opening_check_state(_opening_session("A", 102.0), SESSION, before, p)
    assert math.isnan(s) and math.isnan(vw)
    views = FakeViews([], _opening_session("A", 102.0))
    v = candidate_values(views, SESSION, NSE.epoch_of(DAY, time(9, 30)), ["A"], p)["A"]
    assert math.isnan(v["cand_oc_range_state"])
    # ...and so the rule does not apply: the symbol is kept
    assert evaluate_rule("opening_check", v, p) is None


def test_opening_check_uses_last_price_when_check_minute_has_no_trade():
    cs = [c for c in _opening_session("A", 102.0) if c.ts_epoch < NSE.epoch_of(DAY, time(9, 45))]
    s, _ = opening_check_state(cs, SESSION, NSE.epoch_of(DAY, time(9, 50)), Params())
    assert range_state_name(s) == "above"


# -- filter rules -------------------------------------------------------------------


def test_rules_pass_fail_and_missing():
    p = Params()
    assert evaluate_rule("rsi", {"cand_rsi": 50.0}, p) is True
    assert evaluate_rule("rsi", {"cand_rsi": 80.0}, p) is False
    assert evaluate_rule("rsi", {"cand_rsi": float("nan")}, p) is None
    assert evaluate_rule("run_up", {"cand_run_up_pct": 12.0}, p) is False
    assert evaluate_rule("volume", {"cand_rel_volume": 0.5}, p) is False
    assert evaluate_rule("close_location", {"cand_close_location": 0.2}, p) is False
    assert evaluate_rule("momentum", {"cand_dist_from_high_pct": -15.0}, p) is False
    assert evaluate_rule("momentum", {"cand_dist_from_high_pct": -5.0}, p) is True
    oc = {"cand_oc_range_state": 1.0, "cand_oc_above_vwap": 0.0}
    assert evaluate_rule("opening_check", oc, p) is False  # above range but below VWAP
    p2 = p.with_overrides({"opening_check.require_above_vwap": "false"})
    assert evaluate_rule("opening_check", oc, p2) is True
    assert failed_rules({"cand_rsi": 80.0, "cand_run_up_pct": 1.0}, ["rsi", "run_up"], p) == ["rsi"]


def test_apply_filters_keeps_order_and_is_a_noop_when_off():
    hot = _daily("HOT", [100.0] * 15 + [100, 105, 110, 115, 120, 125])     # +25% in 5 days
    calm = _daily("CALM", [100.0] * 21)
    views = FakeViews(hot + calm, [])
    ranked = [Pick("HOT"), Pick("CALM")]
    as_of = NSE.epoch_of(DAY, time(9, 30))
    assert [x.symbol for x in apply_filters(views, SESSION, as_of, ranked, 5, Params())] == ["HOT", "CALM"]
    assert views.daily_calls == []  # no filter on -> no data touched
    on = Params().with_overrides({"run_up.filter": "true"})
    assert [x.symbol for x in apply_filters(views, SESSION, as_of, ranked, 5, on)] == ["CALM"]


# -- metrics / score order ------------------------------------------------------------


def test_adverse_rate():
    picks = [Pick("A"), Pick("B"), Pick("C"), Pick("D")]
    adverse = {"A": 0.5, "B": 2.0, "C": 3.1}  # D unknown -> not counted
    assert adverse_rate(picks, adverse, k=4, threshold_pct=2.0) == pytest.approx(2 / 3)
    assert "adverse_rate" not in evaluate(picks, [], 4)
    assert evaluate(picks, [], 4, adverse=adverse)["adverse_rate"] == pytest.approx(2 / 3)


def test_rank_by_risk_adjusted():
    pm = [
        PerMinutePrediction("WILD", 1, p_up=1.0, gain_p25=0, gain_p50=2.0, gain_p75=3),
        PerMinutePrediction("STEADY", 1, p_up=1.0, gain_p25=0, gain_p50=1.5, gain_p75=2),
    ]
    atr = {"WILD": 4.0, "STEADY": 1.0}
    assert [s.symbol for s in aggregate_to_topn(pm, 2)] == ["WILD", "STEADY"]
    risk = aggregate_to_topn(pm, 2, rank_by="risk_adjusted", atr_pct=atr)
    assert [s.symbol for s in risk] == ["STEADY", "WILD"]
    assert risk[1].score == pytest.approx(2.0)  # the score itself is not rescaled
    with pytest.raises(ValueError):
        aggregate_to_topn(pm, 2, rank_by="nope")


# -- against stock-analyzer's real data -----------------------------------------------

from tests.conftest import requires_analyzer  # noqa: E402


@requires_analyzer
def test_completed_sessions_have_final_labels(views, sessions):
    every = views.sessions()
    assert sessions and set(sessions) <= set(every)
    for sd in (sessions[0], sessions[-1]):
        assert views.opportunities(sd, source="final"), sd
    # a session with only live labels (today's, while it runs) is never "completed"
    for sd in set(every) - set(sessions):
        assert not views.opportunities(sd, source="final")


@requires_analyzer
def test_candidate_features_enter_the_frame_only_when_switched_on(views, sessions):
    from stock_predictor.features.pipeline import build_feature_frame

    sd = sessions[-25]
    as_of = NSE.epoch_of(date.fromisoformat(sd), time(10, 0))
    off = build_feature_frame(views, sd, as_of, symbols=views.universe(sd)[:6], params=Params())
    assert not [c for c in off.columns if c.startswith("cand_")]
    assert "pct_of_52w_range" not in off.columns  # snapshot-only: removed (look-ahead)
    on = Params().with_overrides({"rsi.feature": "true", "opening_check.feature": "true"})
    f = build_feature_frame(views, sd, as_of, symbols=views.universe(sd)[:6], params=on)
    assert {"cand_rsi", "cand_oc_range_state", "cand_oc_above_vwap"} <= set(f.columns)
    check = NSE.epoch_of(date.fromisoformat(sd), on.opening_check.check_clock)
    assert f.loc[f.ts_epoch < check, "cand_oc_range_state"].isna().all()
    assert f.loc[f.ts_epoch >= check, "cand_oc_range_state"].notna().any()


@requires_analyzer
def test_ablation_smoke(views, sessions):
    from stock_predictor.eval.ablation import format_report, run_ablation

    r = run_ablation(views, sessions[30:34], clock_time=time(9, 45), params=Params())
    assert r.sessions == 4
    assert {x.name for x in r.inputs} == set(Params().candidates())
    assert len(r.quintiles) == 5
    assert "score order" in format_report(r)
