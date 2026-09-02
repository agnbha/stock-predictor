from __future__ import annotations

from stock_predictor.models.aggregate import PerMinutePrediction, aggregate_to_topn


def _pm(symbol, ts, p_up, g50, mtb=30):
    return PerMinutePrediction(symbol, ts, p_up, g50 * 0.6, g50, g50 * 1.4, mtb)


def test_ranks_by_discounted_move_and_takes_top_n():
    pm = [
        _pm("A", 100, 0.8, 2.0), _pm("A", 160, 0.4, 1.0),
        _pm("B", 100, 0.9, 3.0),
        _pm("C", 100, 0.2, 1.0),
        _pm("D", 100, 0.5, 0.5),
    ]
    top = aggregate_to_topn(pm, top_n=2)
    assert [s.symbol for s in top] == ["B", "A"]
    assert top[0].score >= top[1].score
    # A's best minute is the 0.8 * 2.0 one, not the later weaker one
    assert abs(top[1].score - 0.8 * 2.0) < 1e-9


def test_band_is_ordered_and_duration_bucketed():
    top = aggregate_to_topn([_pm("A", 100, 0.7, 1.5, mtb=5)], top_n=5)
    s = top[0]
    assert s.gain_p25 <= s.gain_p50 <= s.gain_p75
    assert s.duration_bucket == "<15m"
    assert s.entry_window_end > s.entry_window_start
