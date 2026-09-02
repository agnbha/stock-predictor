"""Triple-barrier labelling for the stage-1 per-candle model (design section 5).

For each minute, look forward H minutes: an upper barrier at +u, a lower at -d,
a time barrier at H. Barriers are scaled by recent ATR so they mean the same
thing in calm and volatile conditions. This is the labelling stock-analyzer's
model contract already describes -- the two share a definition.

``triple_barrier_labels`` works on one symbol's ordered close/high/low/atr
series and returns, per row: the outcome class, the realised forward return %,
and minutes-to-first-barrier (for the duration bucket).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class TripleBarrierConfig:
    horizon_minutes: int = 30
    upper_atr_mult: float = 1.0
    lower_atr_mult: float = 1.0
    min_barrier_pct: float = 0.15  # floor so a dead-calm ATR still needs a real move


@dataclass(frozen=True)
class BarrierOutcome:
    index: int
    label: int              # +1 upper hit first, -1 lower hit first, 0 time barrier
    forward_return_pct: float
    minutes_to_barrier: int


def triple_barrier_labels(
    close: Sequence[float],
    high: Sequence[float],
    low: Sequence[float],
    atr_pct: Sequence[float],
    cfg: Optional[TripleBarrierConfig] = None,
) -> list[BarrierOutcome]:
    c = cfg or TripleBarrierConfig()
    n = len(close)
    out: list[BarrierOutcome] = []
    for i in range(n):
        entry = close[i]
        if entry <= 0:
            out.append(BarrierOutcome(i, 0, 0.0, c.horizon_minutes))
            continue
        band = max(atr_pct[i] if i < len(atr_pct) else 0.0, c.min_barrier_pct)
        up = entry * (1 + c.upper_atr_mult * band / 100.0)
        dn = entry * (1 - c.lower_atr_mult * band / 100.0)
        end = min(i + c.horizon_minutes, n - 1)
        label, mins = 0, end - i
        for j in range(i + 1, end + 1):
            if high[j] >= up:
                label, mins = 1, j - i
                break
            if low[j] <= dn:
                label, mins = -1, j - i
                break
        fwd = (close[end] - entry) / entry * 100.0
        out.append(BarrierOutcome(i, label, fwd, mins))
    return out
