"""Stage 2 -- aggregate stage-1 output into the daily answer (design section 5).

Mostly a deterministic reduction, not a second learned model:

    for each symbol, at prediction point T:
        expected best window = max over remaining minutes of the forward model's
                               predicted move, discounted by its own probability
        rank the universe by that number, take five

Magnitude and duration come from the same stage-1 outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from stock_predictor.config import DURATION_BUCKETS, duration_bucket


@dataclass(frozen=True)
class SymbolPrediction:
    symbol: str
    score: float                 # discounted expected move, max over remaining minutes
    gain_p25: float
    gain_p50: float
    gain_p75: float
    duration_bucket: str
    entry_window_start: Optional[int] = None
    entry_window_end: Optional[int] = None


@dataclass(frozen=True)
class PerMinutePrediction:
    symbol: str
    ts_epoch: int
    p_up: float
    gain_p25: float
    gain_p50: float
    gain_p75: float
    minutes_to_barrier: Optional[int] = None

    @property
    def discounted_move(self) -> float:
        return max(self.gain_p50, 0.0) * self.p_up


def aggregate_to_topn(
    per_minute: Sequence[PerMinutePrediction],
    top_n: int = 5,
    entry_window_minutes: int = 15,
) -> list[SymbolPrediction]:
    by_symbol: dict[str, list[PerMinutePrediction]] = {}
    for p in per_minute:
        by_symbol.setdefault(p.symbol, []).append(p)

    preds: list[SymbolPrediction] = []
    for sym, rows in by_symbol.items():
        rows = sorted(rows, key=lambda r: r.ts_epoch)
        best = max(rows, key=lambda r: r.discounted_move)
        # gain band: quantiles of the predicted move across the remaining minutes,
        # anchored on the best minute's own band.
        p50s = sorted(r.gain_p50 for r in rows)
        dur_minutes = best.minutes_to_barrier if best.minutes_to_barrier is not None else \
            _modal_duration(rows)
        preds.append(
            SymbolPrediction(
                symbol=sym,
                score=best.discounted_move,
                gain_p25=min(best.gain_p25, p50s[0]),
                gain_p50=best.gain_p50,
                gain_p75=max(best.gain_p75, p50s[-1]),
                duration_bucket=duration_bucket(dur_minutes),
                entry_window_start=best.ts_epoch,
                entry_window_end=best.ts_epoch + entry_window_minutes * 60,
            )
        )
    preds.sort(key=lambda s: s.score, reverse=True)
    return preds[:top_n]


def _modal_duration(rows: Sequence[PerMinutePrediction]) -> float:
    counts = {b: 0 for b in DURATION_BUCKETS}
    for r in rows:
        if r.minutes_to_barrier is not None:
            counts[duration_bucket(r.minutes_to_barrier)] += 1
    if not any(counts.values()):
        return 90.0
    best_bucket = max(counts, key=counts.get)
    return {"<15m": 10.0, "15-60m": 40.0, "60-180m": 120.0, ">180m": 210.0}[best_bucket]
