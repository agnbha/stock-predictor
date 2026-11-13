"""Triple-barrier labelling for the stage-1 per-candle model (design section 5).

For each minute, look forward H minutes: an upper barrier at +u, a lower at -d,
a time barrier at H. By default the barriers are scaled by recent ATR so they
mean the same thing in calm and volatile conditions. This is the labelling
stock-analyzer's model contract already describes -- the two share a definition.

A fixed target instead asks a plain question: does the price touch
+``target_pct`` within the next H minutes? ``stop_pct`` adds a lower barrier
at -``stop_pct``, so the label becomes "target before stop". ``to_close``
replaces the H minutes with the rest of the session: does the price touch
+``target_pct`` at any point before the close?

Every mode also reports ``max_gain_pct``: the best high in the window against
the entry, the most a buyer at that close could have sold for.

With a window to the close, ``outcome`` says what counts as reaching the target:

* ``touch`` (the default): the high reaches +``target_pct``;
* ``close``: the session closes +``target_pct`` or more above the entry;
* ``opportunities``: at least ``count`` separate +``target_pct`` opportunities
  start after the entry. An opportunity is the one stock-analyzer's detector
  marks (``topk-nonoverlap/highlow/.../hold5``): buy at a candle's low, sell at
  the high of a candle at least 5 minutes later; the next one buys after that
  sale. The count is the most such round trips the rest of the session allowed
  (``opportunity_counts``), with perfect hindsight on the timing.

Entry is the close of the row's own candle; the window is the candles after
it whose start is within H clock minutes (a minute with no trades is a gap,
not a shorter window). When one candle reaches both barriers, the order inside
it is unknown and the row counts as stopped. A row whose window runs past the
session's last candle is marked incomplete: its outcome is cut short, so it is
not trained on.

``barrier_arrays`` labels one symbol's ordered series at once;
``triple_barrier_labels`` is the same as a list of outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

OUTCOMES = ("touch", "close", "opportunities")
# As stock-analyzer's opportunity detector: a sale at least this long after the buy.
OPPORTUNITY_HOLD_MINUTES = 5


@dataclass(frozen=True)
class TripleBarrierConfig:
    horizon_minutes: int = 30
    upper_atr_mult: float = 1.0
    lower_atr_mult: float = 1.0
    min_barrier_pct: float = 0.15  # floor so a dead-calm ATR still needs a real move
    # Fixed barriers instead of ATR-scaled ones (the ATR settings are then
    # unused): +target_pct within the horizon; stop_pct adds a lower barrier.
    target_pct: Optional[float] = None
    stop_pct: Optional[float] = None
    # The window is the rest of the session (horizon_minutes is then unused).
    # Needs a fixed target and no stop.
    to_close: bool = False
    # What reaching the target means in a window to the close (module docstring),
    # and for "opportunities", how many.
    outcome: str = "touch"
    count: int = 1

    def __post_init__(self) -> None:
        if self.to_close and (self.target_pct is None or self.stop_pct is not None):
            raise ValueError("to_close needs a fixed target (target_pct) and no stop")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {', '.join(OUTCOMES)}")
        if self.outcome != "touch" and not self.to_close:
            raise ValueError(f"outcome {self.outcome!r} needs a window to the close (to_close)")
        if self.count < 1 or (self.count != 1 and self.outcome != "opportunities"):
            raise ValueError("count is how many opportunities: 1 or more, and only with that outcome")

    @property
    def fixed(self) -> bool:
        return self.target_pct is not None

    def describe(self) -> str:
        if self.to_close:
            if self.outcome == "close":
                return f"a close +{self.target_pct:g}% or more above the entry"
            if self.outcome == "opportunities":
                return f"{self.count}+ separate +{self.target_pct:g}% opportunities before the close"
            return f"+{self.target_pct:g}% before the close"
        h = f"{self.horizon_minutes} min"
        if not self.fixed:
            return f"+{self.upper_atr_mult:g}/-{self.lower_atr_mult:g} x ATR within {h}"
        stop = f" before -{self.stop_pct:g}%" if self.stop_pct is not None else ""
        return f"+{self.target_pct:g}%{stop} within {h}"


@dataclass(frozen=True)
class BarrierOutcome:
    index: int
    label: int              # +1 upper hit first, -1 lower hit first, 0 time barrier
    forward_return_pct: float
    minutes_to_barrier: int
    complete: bool = True   # the whole window lies inside the session


def barrier_arrays(
    close: Sequence[float],
    high: Sequence[float],
    low: Sequence[float],
    atr_pct: Optional[Sequence[float]],
    cfg: Optional[TripleBarrierConfig] = None,
    ts: Optional[Sequence[int]] = None,
    last_ts: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Labels for one symbol's candles, in order.

    ``ts`` (epoch seconds) makes the window H clock minutes; without it, H
    candles. ``last_ts`` is the start of the session's last minute: a window
    ending after it is incomplete. Without it, the last candle given.
    Returns arrays ``label`` (int8), ``forward_return_pct`` (close at the end of
    the window, vs entry), ``max_gain_pct`` (best high in the window, vs entry;
    0 with nothing after), ``minutes_to_barrier`` and ``complete``.
    """
    c = cfg or TripleBarrierConfig()
    cl = np.asarray(close, dtype=np.float64)
    hi = np.asarray(high, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    n = len(cl)
    idx = np.arange(n)
    t = np.asarray(ts, dtype=np.int64) if ts is not None else idx * 60
    if c.to_close:
        return _to_close(cl, hi, lo, t, c)
    limit = t + c.horizon_minutes * 60
    if c.fixed:
        up = cl * (1 + c.target_pct / 100.0)
        dn = cl * (1 - c.stop_pct / 100.0) if c.stop_pct is not None else np.full(n, -np.inf)
    else:
        atr = np.zeros(n) if atr_pct is None else np.nan_to_num(np.asarray(atr_pct, dtype=np.float64))
        band = np.maximum(atr, c.min_barrier_pct)
        up = cl * (1 + c.upper_atr_mult * band / 100.0)
        dn = cl * (1 - c.lower_atr_mult * band / 100.0)

    first_up = np.full(n, n)  # index of the first candle to reach each barrier; n = none
    first_dn = np.full(n, n)
    end = idx.copy()
    best = cl.copy()
    for k in range(1, c.horizon_minutes + 1):
        j = idx + k
        ok = j < n
        j = np.where(ok, j, n - 1)
        ok &= t[j] <= limit
        if not ok.any():
            break
        m = ok & (first_up == n) & (hi[j] >= up)
        first_up[m] = j[m]
        m = ok & (first_dn == n) & (lo[j] <= dn)
        first_dn[m] = j[m]
        end[ok] = j[ok]
        best = np.where(ok, np.maximum(best, hi[j]), best)

    stopped = (first_dn < n) & (first_dn <= first_up)
    hit = (first_up < n) & ~stopped
    label = np.where(stopped, -1, np.where(hit, 1, 0)).astype(np.int8)
    at = np.where(stopped, first_dn, np.where(hit, first_up, end))
    minutes = (t[at] - t) // 60
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd = (cl[end] - cl) / cl * 100.0
        gain = (best - cl) / cl * 100.0
    dead = cl <= 0
    label[dead], fwd[dead], minutes[dead], gain[dead] = 0, 0.0, c.horizon_minutes, 0.0
    complete = limit <= (last_ts if last_ts is not None else (t[-1] if n else 0))
    return {"label": label, "forward_return_pct": fwd, "max_gain_pct": gain,
            "minutes_to_barrier": minutes, "complete": complete}


def first_reach(high: np.ndarray, level: np.ndarray) -> np.ndarray:
    """For each i, the first j > i with ``high[j] >= level[i]``; ``len(high)``
    where none does. One pass from the right with a monotonic stack: of the
    candles after i, only those taller than every nearer one can be first."""
    n = len(high)
    out = np.full(n, n)
    stack_idx: list[int] = []   # nearest candidate last
    stack_neg: list[float] = []  # -high, increasing along the list
    from bisect import bisect_right

    for i in range(n - 2, -1, -1):
        h = float(high[i + 1])
        while stack_neg and -stack_neg[-1] <= h:  # farther and no taller: never first
            stack_neg.pop()
            stack_idx.pop()
        stack_idx.append(i + 1)
        stack_neg.append(-h)
        p = bisect_right(stack_neg, -float(level[i]))  # how many candidates reach the level
        if p:
            out[i] = stack_idx[p - 1]
    return out


def opportunity_counts(high, low, ts, pct: float,
                       hold_minutes: int = OPPORTUNITY_HOLD_MINUTES) -> np.ndarray:
    """``out[s]`` for s in 0..n: the most separate opportunities among candles
    s onwards (``out[n]`` is 0). An opportunity buys at a candle's low and sells
    at the high of a candle at least ``hold_minutes`` later, ``pct``% or more
    above; the next buys at a candle after that sale.

    Taking, each time, the opportunity whose sale comes first leaves the most
    room for the rest, so it gives the most (interval scheduling)."""
    hi = np.asarray(high, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    t = np.asarray(ts, dtype=np.int64)
    n = len(hi)
    out = np.zeros(n + 1, dtype=np.int64)
    if n < 2:
        return out
    level = np.where(lo > 0, lo * (1 + pct / 100.0), np.inf)
    earliest = np.searchsorted(t, t + hold_minutes * 60, side="left")  # first candle a sale may be
    j = np.arange(n)
    ok = (hi[None, :] >= level[:, None]) & (j[None, :] >= earliest[:, None])
    sale = np.where(ok.any(axis=1), ok.argmax(axis=1), n)  # the first sale for a buy at each candle
    first = np.minimum.accumulate(sale[::-1])[::-1]        # ... for a buy at that candle or later
    for s in range(n - 1, -1, -1):
        out[s] = 0 if first[s] >= n else 1 + out[first[s] + 1]
    return out


def _to_close(cl: np.ndarray, hi: np.ndarray, lo: np.ndarray, t: np.ndarray,
              cfg: TripleBarrierConfig) -> dict:
    """The window is every later candle in the session; it always ends at the
    close, so every row is complete."""
    target_pct = cfg.target_pct
    n = len(cl)
    up = cl * (1 + target_pct / 100.0)
    first = first_reach(hi, up)
    hit = first < n
    label = hit.astype(np.int8)
    after = np.empty(n)
    if n:
        after[:-1] = np.maximum.accumulate(hi[::-1])[::-1][1:]  # best high strictly later
        after[-1] = cl[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd = (cl[-1] - cl) / cl * 100.0 if n else np.zeros(0)
        gain = np.maximum(after - cl, 0.0) / cl * 100.0
    last = t[-1] if n else 0
    minutes = np.where(hit, (t[np.minimum(first, n - 1)] - t) // 60, (last - t) // 60)
    out = {}
    if cfg.outcome == "close":
        label = (fwd >= target_pct - 1e-9).astype(np.int8)
        minutes = (last - t) // 60
    elif cfg.outcome == "opportunities":
        count = opportunity_counts(hi, lo, t, target_pct)[1:]  # buys after each row's own candle
        label = (count >= cfg.count).astype(np.int8)
        minutes = (last - t) // 60
        out["opportunities"] = count
    dead = cl <= 0
    label[dead], fwd[dead], gain[dead] = 0, 0.0, 0.0
    return {"label": label, "forward_return_pct": fwd, "max_gain_pct": gain,
            "minutes_to_barrier": minutes, "complete": np.ones(n, dtype=bool), **out}


def triple_barrier_labels(
    close: Sequence[float],
    high: Sequence[float],
    low: Sequence[float],
    atr_pct: Sequence[float],
    cfg: Optional[TripleBarrierConfig] = None,
    ts: Optional[Sequence[int]] = None,
    last_ts: Optional[int] = None,
) -> list[BarrierOutcome]:
    a = barrier_arrays(close, high, low, atr_pct, cfg, ts, last_ts)
    return [
        BarrierOutcome(i, int(a["label"][i]), float(a["forward_return_pct"][i]),
                       int(a["minutes_to_barrier"][i]), bool(a["complete"][i]))
        for i in range(len(a["label"]))
    ]
