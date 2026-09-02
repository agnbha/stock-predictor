from __future__ import annotations

from stock_predictor.eval.metrics import (
    Pick,
    duration_accuracy,
    evaluate,
    ndcg_at_k,
    precision_at_k,
    quantile_coverage,
    realised_net_gain,
)
from stock_predictor.labels.targets import Target


def _t(symbol, rank, gain, dur=30, bucket="15-60m"):
    return Target(
        session_date="2026-07-01", as_of=0, symbol=symbol, universe_rank=rank,
        entry_ts=100, exit_ts=200, gain_pct=gain, duration_minutes=dur,
        duration_bucket=bucket, in_top_n=rank <= 5,
    )


TRUTH = [_t("A", 1, 2.0), _t("B", 2, 1.5), _t("C", 3, 1.0), _t("D", 4, 0.8), _t("E", 5, 0.5),
         _t("F", 6, 0.2)]


def test_precision_at_k():
    picks = [Pick("A"), Pick("B"), Pick("X"), Pick("Y"), Pick("Z")]
    assert precision_at_k(picks, TRUTH, k=5) == 2 / 5
    assert precision_at_k([Pick(t.symbol) for t in TRUTH[:5]], TRUTH, k=5) == 1.0


def test_ndcg_rewards_order():
    good = [Pick(t.symbol) for t in TRUTH[:5]]
    bad = [Pick(t.symbol) for t in reversed(TRUTH[:5])]
    assert ndcg_at_k(good, TRUTH, 5) > ndcg_at_k(bad, TRUTH, 5)
    assert ndcg_at_k(good, TRUTH, 5) == 1.0


def test_realised_net_gain_is_net_of_costs():
    picks = [Pick("A"), Pick("B")]
    g = realised_net_gain(picks, TRUTH, k=2)
    assert g < (2.0 + 1.5) / 2  # costs shaved off
    assert realised_net_gain([Pick("missing")], TRUTH, k=1) == 0.0


def test_quantile_coverage_and_duration_accuracy():
    picks = [Pick("A", gain_p25=1.0, gain_p75=3.0, duration_bucket="15-60m"),
             Pick("B", gain_p25=2.0, gain_p75=3.0, duration_bucket=">180m")]
    assert quantile_coverage(picks, TRUTH, 2) == 0.5  # A covers 2.0, B misses 1.5
    assert duration_accuracy(picks, TRUTH, 2) == 0.5


def test_evaluate_returns_all_keys():
    m = evaluate([Pick("A"), Pick("B")], TRUTH, k=5)
    assert {"realised_net_gain", "precision_at_k", "ndcg_at_k", "quantile_coverage",
            "duration_accuracy", "hits"} <= m.keys()
