"""Metrics, in the order they matter (design section 6).

1. Realised net gain of the five picks, after costs -- gross is close to
   meaningless here.
2. Precision@5 and NDCG@5 -- did the picks land in the day's genuine top set.
3. Quantile coverage -- do ~50% of outcomes fall inside the p25-p75 band.
4. Duration accuracy by bucket.
5. Adverse rate -- share of picks that fell at least ``adverse_move_pct`` below
   their as-of price later that session. A ranking that wins on gain only by
   picking more volatile symbols shows up here (design 4.1, score order).

A "prediction" here is an ordered list of :class:`Pick`. "Truth" is the
:class:`~stock_predictor.labels.targets.Target` list for the same
``(session, as_of)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from stock_predictor.eval.charges import ChargeRates, net_gain_pct
from stock_predictor.labels.targets import Target


@dataclass(frozen=True)
class Pick:
    symbol: str
    gain_p25: float = 0.0
    gain_p50: float = 0.0
    gain_p75: float = 0.0
    duration_bucket: Optional[str] = None
    entry_price: float = 0.0  # for net-of-cost; 0 -> use a nominal 0.075% haircut


def _truth_by_symbol(truth: Sequence[Target]) -> dict[str, Target]:
    return {t.symbol: t for t in truth}


def _realised_net(sym: str, by_sym: dict[str, Target], rates: Optional[ChargeRates]) -> float:
    t = by_sym.get(sym)
    if t is None:
        return 0.0
    # Target carries gross gain_pct; entry price is not on it, so use the
    # notional-based cost model with a representative price when unknown.
    return net_gain_pct(t.gain_pct, entry_price=1000.0, rates=rates)


def precision_at_k(picks: Sequence[Pick], truth: Sequence[Target], k: int = 5) -> float:
    true_top = {t.symbol for t in truth if t.universe_rank <= k}
    if not true_top:
        return 0.0
    picked = [p.symbol for p in picks[:k]]
    hits = sum(1 for s in picked if s in true_top)
    return hits / k


def ndcg_at_k(picks: Sequence[Pick], truth: Sequence[Target], k: int = 5,
              rates: Optional[ChargeRates] = None) -> float:
    by_sym = _truth_by_symbol(truth)
    rels = [max(_realised_net(p.symbol, by_sym, rates), 0.0) for p in picks[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(
        (max(net_gain_pct(t.gain_pct, 1000.0, rates=rates), 0.0) for t in truth), reverse=True
    )[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def realised_net_gain(picks: Sequence[Pick], truth: Sequence[Target], k: int = 5,
                      rates: Optional[ChargeRates] = None) -> float:
    """Mean net-of-cost gain of the top-k picks. The number that matters."""
    by_sym = _truth_by_symbol(truth)
    top = picks[:k]
    if not top:
        return 0.0
    return sum(_realised_net(p.symbol, by_sym, rates) for p in top) / len(top)


def hit_count(picks: Sequence[Pick], truth: Sequence[Target], k: int = 5,
              min_net_pct: float = 0.0) -> int:
    """How many of the k picks actually cleared ``min_net_pct`` net."""
    by_sym = _truth_by_symbol(truth)
    return sum(1 for p in picks[:k] if _realised_net(p.symbol, by_sym, None) > min_net_pct)


def quantile_coverage(picks: Sequence[Pick], truth: Sequence[Target], k: int = 5) -> float:
    """Fraction of realised gains inside the pick's p25-p75 band. Target ~0.5;
    a miscalibrated band is worse than no band."""
    by_sym = _truth_by_symbol(truth)
    inside = total = 0
    for p in picks[:k]:
        t = by_sym.get(p.symbol)
        if t is None:
            continue
        total += 1
        if p.gain_p25 <= t.gain_pct <= p.gain_p75:
            inside += 1
    return inside / total if total else 0.0


def duration_accuracy(picks: Sequence[Pick], truth: Sequence[Target], k: int = 5) -> float:
    by_sym = _truth_by_symbol(truth)
    correct = total = 0
    for p in picks[:k]:
        t = by_sym.get(p.symbol)
        if t is None or p.duration_bucket is None:
            continue
        total += 1
        if p.duration_bucket == t.duration_bucket:
            correct += 1
    return correct / total if total else 0.0


def adverse_rate(picks: Sequence[Pick], adverse: dict[str, float], k: int = 5,
                 threshold_pct: float = 2.0) -> float:
    """Share of the top-k picks whose price later fell ``threshold_pct`` or more
    below the as-of price. ``adverse`` comes from
    :func:`~stock_predictor.labels.targets.adverse_excursions`."""
    scored = [adverse[p.symbol] for p in picks[:k] if p.symbol in adverse]
    if not scored:
        return 0.0
    return sum(1 for a in scored if a >= threshold_pct) / len(scored)


def evaluate(picks: Sequence[Pick], truth: Sequence[Target], k: int = 5,
             rates: Optional[ChargeRates] = None,
             adverse: Optional[dict[str, float]] = None,
             adverse_move_pct: float = 2.0) -> dict[str, float]:
    out = {
        "realised_net_gain": realised_net_gain(picks, truth, k, rates),
        "precision_at_k": precision_at_k(picks, truth, k),
        "ndcg_at_k": ndcg_at_k(picks, truth, k, rates),
        "quantile_coverage": quantile_coverage(picks, truth, k),
        "duration_accuracy": duration_accuracy(picks, truth, k),
        "hits": float(hit_count(picks, truth, k)),
    }
    if adverse is not None:
        out["adverse_rate"] = adverse_rate(picks, adverse, k, adverse_move_pct)
    return out


def aggregate(per_session: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across sessions."""
    if not per_session:
        return {}
    keys = per_session[0].keys()
    return {key: sum(d[key] for d in per_session) / len(per_session) for key in keys}
