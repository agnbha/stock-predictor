from __future__ import annotations

from datetime import date

from stock_predictor.clock import NSE
from stock_predictor.config import duration_bucket
from stock_predictor.data.store import PredictionRow, Store
from stock_predictor.fundamentals import CsvFundamentalScore, NullFundamentalScore
from stock_predictor.models.triple_barrier import TripleBarrierConfig, triple_barrier_labels
from stock_predictor.news import NullNewsSentiment


def test_clock_matches_the_sql_session_date():
    # v_intraday_candles derives session_date as +5h30m from the epoch
    assert NSE.session_length_minutes() == 375
    assert NSE.minutes_since_open(NSE.epoch_of(date(2026, 8, 27), NSE.open_time)) == 0


def test_duration_buckets():
    assert duration_bucket(5) == "<15m"
    assert duration_bucket(30) == "15-60m"
    assert duration_bucket(120) == "60-180m"
    assert duration_bucket(400) == ">180m"


def test_triple_barrier_hits_upper_then_time():
    close = [100, 100, 100, 100, 100]
    high = [100, 100, 103, 100, 100]
    low = [100, 99, 99, 99, 99]
    atr = [2, 2, 2, 2, 2]
    out = triple_barrier_labels(close, high, low, atr, TripleBarrierConfig(horizon_minutes=3))
    assert out[0].label == 1 and out[0].minutes_to_barrier == 2
    assert out[4].label == 0  # nothing ahead -> time barrier


def test_store_roundtrip_and_scoring(tmp_path):
    s = Store(tmp_path / "s.db")
    row = PredictionRow(model_version="m/1", session_date="2026-07-01", as_of=1, computed_at=1,
                        rank=1, symbol="A", gain_p50=1.2)
    s.save_predictions([row])
    latest = s.latest_prediction("m/1")
    assert latest and latest[0]["symbol"] == "A"
    pending = s.unscored_predictions(before_session="2026-07-02")
    assert len(pending) == 1
    s.score_prediction(pending[0]["id"], realized_gain_pct=0.9, realized_duration_minutes=20, scored_at=2)
    assert not s.unscored_predictions(before_session="2026-07-02")
    s.close()


def test_fundamentals_and_news_null_impls():
    assert NullFundamentalScore().score("RELIANCE", date(2026, 8, 1)) is None
    assert NullNewsSentiment().sentiment("RELIANCE", 1787802300) is None


def test_csv_fundamentals_is_point_in_time(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("symbol,effective_date,score\nRELIANCE,2026-04-01,70\nRELIANCE,2026-07-01,80\n")
    fs = CsvFundamentalScore(p)
    assert fs.score("RELIANCE", date(2026, 5, 1)) == 70.0
    assert fs.score("RELIANCE", date(2026, 8, 1)) == 80.0
    assert fs.score("RELIANCE", date(2026, 1, 1)) is None
