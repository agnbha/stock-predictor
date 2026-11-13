"""The on-demand summary (stock-predictor summary, scripts/today.sh)."""

from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from datetime import date, time

from stock_predictor.clock import NSE
from stock_predictor.models.registry import ModelCard, ModelRegistry
from stock_predictor.models.triple_barrier import TripleBarrierConfig
from stock_predictor.serving.summary import build_summary, find_model, format_summary, model_state

CFG = TripleBarrierConfig(target_pct=0.5, to_close=True)
AT_1030 = NSE.epoch_of(date(2026, 11, 11), time(10, 30))
ROOT = Path(__file__).resolve().parents[1]


def _rows():
    return pd.DataFrame({
        "symbol": ["A", "B", "C", "D", "E", "F"],
        "price": [100.0, 200.0, 50.0, 10.0, 80.0, 30.0],
        "chance": [0.70, 0.66, 0.61, 0.58, 0.52, 0.40],
        "p_model": [0.8, 0.7, 0.6, 0.55, 0.5, 0.3],
        "p_seen": [0.70, 0.66, 0.61, 0.58, 0.52, 0.40],
        "p25": [0.55, 0.30, 0.20, 0.25, 0.10, 0.05],
        "p50": [1.20, 0.95, 0.74, 0.71, 0.52, 0.30],
        "p75": [2.0, 1.5, 1.2, 1.1, 0.9, 0.6],
        "vol_rank": [3, 1, 2, 4, 5, 6],
    })


EVAL = {
    "model_version": "m/2026-10-06", "sessions": ["2026-10-09", "2026-11-09"], "session_count": 21,
    "cost_pct": 0.083,
    "summary": {"base_hit": 0.55, "base_net": 0.03, "model": {"hit": 0.69, "net": 0.035},
                "intraday_volatility": {"hit": 0.69, "net": -0.02}},
    "by_point": [{"point": "09:30", "base_hit": 0.74}, {"point": "10:30", "base_hit": 0.59},
                 {"point": "11:00", "base_hit": 0.6}],
    "filters": [{"rule": "chance >= 50% (calibrated)", "per_session_point": 2.2, "hit": 0.75}],
    "band_coverage": {"band": "max_gain_pct", "p25": 0.21, "p50": 0.40, "p75": 0.67},
}


def test_summary_filters_lists_groups_and_explains():
    card = SimpleNamespace(trained_through="2026-11-05", train_sessions=100)
    s = build_summary(_rows(), CFG, "m/2026-11-05", card, "2026-11-11", as_of=AT_1030,
                      finished=AT_1030, latest_session="2026-11-09", evaluation=EVAL,
                      min_chance=0.5, top=2, now=AT_1030 + 60 * 8)
    assert (s["universe"], s["passed"]) == (6, 5)  # F's 40% is below the bar
    assert [r["symbol"] for r in s["top"]] == ["A", "B"]
    assert s["top"][0]["target_price"] == pytest.approx(100.5)
    assert [(g["from_pct"], g["symbols"]) for g in s["rest"]["groups"]] == [(0.7, ["C", "D"]), (0.5, ["E"])]
    assert s["rest"]["chance_range"] == [0.52, 0.61]
    assert s["standouts"] == {"whole_range_above_target": ["A"], "median_twice_target": ["A"]}
    assert s["trust"]["base_hit_now"] == 0.59 and s["trust"]["point"] == "10:30"  # as_of is 10:30
    assert s["trust"]["beat_median"] == pytest.approx(0.60)
    notes = " ".join(s["notes"])
    assert "trained through 2026-11-05; 2026-11-09 has finished" in notes
    assert "7 minutes behind" in notes

    assert s["top"][1]["range_price"] == pytest.approx([200.60, 203.00])  # B: 200 raised by +0.30% and +1.5%
    # entry: 0.6% below the low end of the range (A: 100.55 -> 99.95), never above the price now
    assert s["top"][0]["entry_price"] == pytest.approx(100.55 * 0.994)
    assert s["top"][1]["entry_price"] == pytest.approx(200.60 * 0.994)
    assert all(r["entry_price"] <= r["price"] for r in s["all"])
    assert all(r["range_price"][0] / r["entry_price"] >= 1 / 0.994 - 1e-9 or r["entry_price"] == r["price"]
               for r in s["all"])
    high_floor = _rows()
    high_floor.loc[0, "p25"] = 1.0  # a floor already 1% up: the margin is there at the price now
    h = build_summary(high_floor, CFG, "m", card, "2026-11-11", as_of=AT_1030, finished=AT_1030,
                      latest_session="2026-11-09", evaluation=None)
    assert h["top"][0]["entry_price"] == 100.0 and h["top"][0]["entry_vs_price_pct"] == 0
    assert "100.00 (now)" in format_summary(h)

    text = format_summary(s)
    assert any(ln.split()[:2] == ["#", "Stock"] and "Range in price" in ln for ln in text.splitlines())
    assert "+0.55 to +2.00%    100.55 to 102.00" in text and "200.60 to 203.00" in text
    assert any(ln.split()[:2] == ["#", "Stock"] and "Entry price" in ln for ln in text.splitlines())
    assert "100.55 to 102.00  99.95 (-0.1%)" in text and "Entry price: the most to pay. It is 0.6% below" in text
    assert "5 of 6 stocks have a 50% or better chance" in text
    assert "+0.70 to +0.79%  C, D" in text
    assert "The model's top 5: 69%; the 5 most volatile stocks: 69% (-0.02% a trade)." in text
    assert "Any stock reached +0.5% 55% of the time (59% from 10:30)." in text


def test_summary_without_an_evaluation_or_for_a_past_session():
    card = SimpleNamespace(trained_through="2026-11-09", train_sessions=110)
    s = build_summary(_rows(), CFG, "m", card, "2026-11-09", as_of=AT_1030, finished=AT_1030,
                      latest_session="2026-11-06", evaluation=None, live=False)
    assert s["trust"] is None and s["lag_minutes"] is None
    notes = " ".join(s["notes"])
    assert "--evaluate" in notes and "in-sample" in notes
    assert "before 2026-11-09's close" in format_summary(s)


def _card(version, trained, target, band="max_gain_pct"):
    return ModelCard(version, "sklearn", trained, 10, ["x"], target=target, band=band)


def test_the_model_for_a_target_and_whether_it_is_current(tmp_path):
    reg = ModelRegistry(tmp_path)
    t05 = asdict(CFG)
    for version, trained, target, band in (
        ("a", "2026-11-05", t05, "max_gain_pct"),
        ("b", "2026-11-09", t05, "max_gain_pct"),
        ("c", "2026-11-10", t05, "forward_return_pct"),  # not a best-gain band
        ("d", "2026-11-10", asdict(TripleBarrierConfig(horizon_minutes=40, target_pct=0.5)), "max_gain_pct"),
    ):
        reg.save(version, None, _card(version, trained, target, band))
    assert find_model(reg, 0.5) == "b"
    assert find_model(reg, 1.0) is None
    views = SimpleNamespace(sessions=lambda: ["2026-11-06", "2026-11-09"])
    assert model_state(0.5, reg, views) == "ok"
    views = SimpleNamespace(sessions=lambda: ["2026-11-09", "2026-11-11"])
    assert model_state(0.5, reg, views) == "stale"
    assert model_state(1.0, reg, views) == "missing"


@pytest.mark.skipif(not (ROOT / ".venv" / "bin" / "stock-predictor").exists(), reason="needs the repo's .venv")
def test_today_script_parses_and_checks_its_arguments():
    script = ROOT / "scripts" / "today.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    help_ = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True)
    assert help_.returncode == 0 and "--retrain" in help_.stdout and "--evaluate" in help_.stdout
    bad = subprocess.run(["bash", str(script), "--target", "abc"], capture_output=True, text=True)
    assert bad.returncode == 2 and "must be a number" in bad.stderr


# -- models trained on only the last few weeks ---------------------------------


def test_recent_window_and_minute_step_are_named_and_kept_apart(tmp_path):
    from stock_predictor.models.registry import model_version_string

    assert model_version_string("sklearn", "d", CFG) == "sklearn-target-0.5pct-close/d"
    assert model_version_string("sklearn", "d", CFG, 3) == "sklearn-target-0.5pct-close-m3/d"
    assert model_version_string("sklearn", "d", CFG, 1, 30) == "sklearn-target-0.5pct-close-recent30d/d"
    assert model_version_string("sklearn", "d", None, 5, 30) == "sklearn-triple-barrier-recent30d-m5/d"

    reg = ModelRegistry(tmp_path)
    t05 = asdict(CFG)
    full = ModelCard("full", "sklearn", "2026-11-09", 110, ["x"], target=t05, band="max_gain_pct", minute_step=3)
    full_twin = ModelCard("full-twin", "sklearn", "2026-10-08", 90, ["x"], target=t05, band="max_gain_pct",
                          minute_step=3)
    recent = ModelCard("recent", "sklearn", "2026-11-09", 20, ["x"], target=t05, band="max_gain_pct",
                       minute_step=1, recent_days=30, trained_from="2026-10-12")
    recent_twin = ModelCard("recent-twin", "sklearn", "2026-10-08", 22, ["x"], target=t05, band="max_gain_pct",
                            minute_step=1, recent_days=30)
    for c in (full_twin, recent_twin, full, recent):
        reg.save(c.model_version, None, c)
    reg.save_extra("full-twin", "evaluation", {"who": "full"})
    reg.save_extra("recent-twin", "evaluation", {"who": "recent"})

    assert find_model(reg, 0.5) == "full" and find_model(reg, 0.5, recent=True) == "recent"
    assert reg.latest_with_target() == "full"  # the dashboard's default stays full history
    # each model finds the held-out check of its own kind
    assert reg.extra_like(full, "evaluation") == ("full-twin", {"who": "full"})
    assert reg.extra_like(recent, "evaluation") == ("recent-twin", {"who": "recent"})
    views = SimpleNamespace(sessions=lambda: ["2026-11-09"])
    assert model_state(0.5, reg, views, recent=True) == "ok"


def test_recent_days_keeps_the_last_calendar_days_of_what_was_chosen():
    from stock_predictor.cli import _sessions

    days = ["2026-09-01", "2026-09-30", "2026-10-01", "2026-10-15", "2026-10-31"] + [
        f"2026-11-{d:02d}" for d in range(1, 22)]
    views = SimpleNamespace(sessions=lambda start=None, end=None: list(days))
    all_ = SimpleNamespace(all_sessions=True, recent_days=30)
    assert _sessions(views, all_, candles_only=True)[0] == "2026-10-31"  # 2026-11-21 minus 30 days
    # development sessions are all but the last 21 (09-01 .. 10-31); their last 30 days:
    dev = SimpleNamespace(recent_days=30)
    assert _sessions(views, dev, candles_only=True) == ["2026-10-15", "2026-10-31"]
    assert _sessions(views, SimpleNamespace(all_sessions=True), candles_only=True) == days
