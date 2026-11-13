"""The next session (tomorrow.sh), recorded picks and the scorecard, and the
daily refresh."""

from __future__ import annotations

import subprocess
from dataclasses import asdict
from datetime import date, time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from stock_predictor.clock import NSE
from stock_predictor.data.analyzer import Candle
from stock_predictor.models.nextday import next_trading_day, session_outcomes, state_at
from stock_predictor.models.registry import ModelCard, ModelRegistry, model_version_string
from stock_predictor.models.triple_barrier import TripleBarrierConfig
from stock_predictor.serving.scorecard import format_scorecard, score_pick, scorecard, track_records
from stock_predictor.serving.summary import find_model

ROOT = Path(__file__).resolve().parents[1]
CFG = TripleBarrierConfig(target_pct=0.5, to_close=True)


def _candles(session, symbol, rows):
    day = date.fromisoformat(session)
    return [Candle(symbol, NSE.epoch_of(day, time(9, 15 + i)), session, *ohlc, 100) for i, ohlc in enumerate(rows)]


class _Views:
    def __init__(self, by_session, finished=()):
        self.by_session, self.finished = by_session, list(finished)

    def intraday_candles(self, session, symbols=None, as_of=None):
        return [c for c in self.by_session.get(session, []) if as_of is None or c.ts_epoch <= as_of]

    def sessions(self, start=None, end=None):
        return self.finished


def test_next_trading_day_skips_weekends_and_holidays(tmp_path):
    (tmp_path / "nse-holidays-2026.txt").write_text("# NSE\n2026-11-10   # Diwali\n")
    assert next_trading_day(date(2026, 11, 9), tmp_path) == (date(2026, 11, 11), True)
    assert next_trading_day(date(2026, 11, 6), tmp_path) == (date(2026, 11, 9), True)  # Friday -> Monday
    assert next_trading_day(date(2026, 12, 31), tmp_path) == (date(2027, 1, 1), False)  # no 2027 list


def test_outcomes_are_measured_from_the_open():
    views = _Views({"2026-11-11": _candles("2026-11-11", "A", [(100, 100.2, 99, 100), (100, 100.6, 99.5, 100.1),
                                                                 (100.1, 100.3, 98, 98.5)])})
    out = session_outcomes(views, "2026-11-11", 0.5).set_index("symbol").loc["A"]
    assert out["open"] == 100 and out["label_class"] == 1
    assert out["max_gain_pct"] == pytest.approx(0.6) and out["close_return_pct"] == pytest.approx(-1.5)
    assert out["session_close"] == 98.5 and out["opportunities"] == 0  # 3 minutes: no 5-minute hold
    frame = pd.DataFrame({"symbol": ["A", "A", "B"], "ts_epoch": [1, 3, 2], "x": [1, 2, 3]})
    assert state_at(frame, 2).set_index("symbol")["x"].to_dict() == {"A": 1, "B": 3}


def test_scoring_a_recorded_pick():
    cs = _candles("2026-11-11", "A", [(100, 100.1, 99.9, 100), (100, 100.7, 99.8, 100.2), (100.2, 100.3, 99, 99.5)])
    today = {"kind": "today", "as_of": cs[0].ts_epoch, "price": 100.0, "target_pct": 0.5, "id": 1}
    o = score_pick(today, cs)
    assert o["hit"] == 1 and o["best_gain_pct"] == pytest.approx(0.7) and o["close_return_pct"] == pytest.approx(-0.5)
    late = dict(today, as_of=cs[1].ts_epoch, price=100.2)  # after the high: only 100.3 is left
    assert score_pick(late, cs)["hit"] == 0
    nxt = {"kind": "next", "as_of": 0, "price": 120.0, "target_pct": 0.5, "id": 2}  # entry is the open, not 120
    assert score_pick(nxt, cs)["entry_price"] == 100 and score_pick(nxt, cs)["hit"] == 1
    assert score_pick(today, []) is None


def test_recorded_picks_are_scored_once_the_session_closes(store):
    day = "2026-11-11"
    cs = (_candles(day, "A", [(100, 100.1, 99.9, 100), (100, 100.9, 99.8, 100.5)])
          + _candles(day, "B", [(50, 50.05, 49.9, 50), (50, 50.1, 49.5, 49.6)]))
    views = _Views({day: cs})
    as_of = cs[0].ts_epoch
    rows = [{"made_at": 1, "kind": "today", "model_version": "m", "target_pct": 0.5, "session_date": day,
             "base_session": day, "as_of": as_of, "symbol": s, "rank": r, "price": p, "chance": c,
             "p_model": c, "gain_p25": 0.2, "gain_p50": 0.6, "gain_p75": 1.0, "passed": int(c >= 0.5)}
            for s, r, p, c in (("A", 1, 100.0, 0.7), ("B", 2, 50.0, 0.4))]
    assert store.save_summary_picks(rows) == 2 and store.save_summary_picks(rows) == 0  # once per minute

    running = scorecard(views, store, day, now=as_of + 600)
    assert not running["closed"] and running["saved"] == 0
    run = running["runs"][0]
    assert (run["passed"], run["passed_hits"], run["top5_hits"]) == (1, 1, 1)
    assert run["every_stock_hit"] == 0.5
    assert "so far" in format_scorecard(running) and "A ✓" in format_scorecard(running)

    after = NSE.session_close_epoch(date.fromisoformat(day)) + 120
    assert scorecard(views, store, day, now=after)["saved"] == 2
    assert scorecard(views, store, day, now=after + 60)["saved"] == 0  # already scored
    rec = track_records(store, "2026-11-01")["today"]
    assert rec["A"] == {"checks": 1, "said_yes": 1, "yes_reached": 1, "right": 1, "reached": 1}
    assert rec["B"] == {"checks": 1, "said_yes": 0, "yes_reached": 0, "right": 1, "reached": 0}


def test_next_session_models_are_named_found_and_pruned_apart(tmp_path):
    assert model_version_string("sklearn", "d", CFG, next_session=True) == "sklearn-nextday-0.5pct/d"
    reg = ModelRegistry(tmp_path)
    t = asdict(CFG)
    for v, through, entry in (("today", "2026-11-09", "minute"), ("next-old", "2026-11-05", "next_open"),
                              ("next-mid", "2026-11-06", "next_open"), ("next", "2026-11-09", "next_open"),
                              ("next-twin", "2026-10-08", "next_open")):
        reg.save(v, None, ModelCard(v, "sklearn", through, 1, ["x"], target=t, band="max_gain_pct",
                                    minute_step=1 if entry == "next_open" else 3, entry=entry))
    reg.save_extra("next-twin", "evaluation", {"x": 1})
    assert find_model(reg, 0.5) == "today"
    assert find_model(reg, 0.5, next_session=True) == "next"
    assert reg.latest_with_target() == "today"  # the dashboard stays on today's model
    # keep the newest 2 of each kind, and the newest holder of a held-out check
    assert reg.prune(keep=2, dry_run=True) == ["next-old"]
    assert reg.prune(keep=2) == ["next-old"] and not (tmp_path / "next-old").exists()
    assert (tmp_path / "next-twin").exists()


@pytest.mark.skipif(not (ROOT / ".venv" / "bin" / "stock-predictor").exists(), reason="needs the repo's .venv")
@pytest.mark.parametrize("script", ["tomorrow.sh", "refresh.sh", "schedule-refresh.sh"])
def test_scripts_parse_and_explain_themselves(script):
    path = ROOT / "scripts" / script
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    out = subprocess.run(["bash", str(path), "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and len(out.stdout.splitlines()) > 3
