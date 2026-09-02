"""Target construction -- and the one place the leakage rule lives.

A prediction is made **as of a moment T**. Design section 2:

    * features may use only data available at or before T;
    * the label counts only windows whose **entry is strictly after T**.

Every label in the project is built through :func:`build_targets`. Nothing else
is allowed to read ``v_opportunity_markers`` for training. That keeps the rule
auditable: :func:`leakage_audit` re-checks it, and ``tests/test_leakage.py``
runs the audit over real sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from stock_predictor.config import CONFIG, duration_bucket
from stock_predictor.data.analyzer import AnalyzerViews, Opportunity


@dataclass(frozen=True)
class Target:
    """One row of the answer: which symbol, which window, how long, how much."""

    session_date: str
    as_of: int
    symbol: str
    universe_rank: int          # 1 = best remaining window in the universe
    entry_ts: int
    exit_ts: int
    gain_pct: float
    duration_minutes: int
    duration_bucket: str
    in_top_n: bool              # is this symbol in the day's genuine top N

    @property
    def entry_after_as_of(self) -> bool:
        return self.entry_ts > self.as_of


def remaining_windows(
    views: AnalyzerViews,
    session_date: str,
    as_of: int,
    detector_version: Optional[str] = None,
    source: str = "final",
) -> list[Opportunity]:
    """Every detector window on ``session_date`` whose entry is strictly after
    ``as_of``. This is the only gate; downstream code never re-filters."""
    return views.opportunities(
        session_date, detector_version=detector_version, as_of=as_of, source=source
    )


def build_targets(
    views: AnalyzerViews,
    session_date: str,
    as_of: int,
    top_n: Optional[int] = None,
    detector_version: Optional[str] = None,
    universe: Optional[Sequence[str]] = None,
    source: str = "final",
) -> list[Target]:
    """The labelled universe for one prediction point.

    Returns one :class:`Target` per symbol that still has a window open after
    ``as_of`` -- its single best remaining window -- ranked by gain. The top
    ``top_n`` are flagged ``in_top_n``; the rest are the negatives a ranker
    needs. Symbols with nothing left after ``as_of`` drop out (correctly: at
    14:30 most of the day's opportunity is already spent).
    """
    n = top_n if top_n is not None else CONFIG.top_n
    windows = remaining_windows(views, session_date, as_of, detector_version, source)
    if universe is not None:
        allowed = set(universe)
        windows = [w for w in windows if w.symbol in allowed]

    best_per_symbol: dict[str, Opportunity] = {}
    for w in windows:
        cur = best_per_symbol.get(w.symbol)
        if cur is None or w.gain_pct > cur.gain_pct:
            best_per_symbol[w.symbol] = w

    ordered = sorted(best_per_symbol.values(), key=lambda w: w.gain_pct, reverse=True)
    targets: list[Target] = []
    for i, w in enumerate(ordered, start=1):
        targets.append(
            Target(
                session_date=session_date,
                as_of=as_of,
                symbol=w.symbol,
                universe_rank=i,
                entry_ts=w.entry_ts,
                exit_ts=w.exit_ts,
                gain_pct=w.gain_pct,
                duration_minutes=w.duration_minutes,
                duration_bucket=duration_bucket(w.duration_minutes),
                in_top_n=i <= n,
            )
        )
    return targets


def gain_band(gains: Sequence[float]) -> tuple[float, float, float]:
    """p25 / p50 / p75 of a set of realised gains -- the honest band the
    magnitude model is trained to reproduce (design section 1)."""
    if not gains:
        return (0.0, 0.0, 0.0)
    s = sorted(gains)

    def q(p: float) -> float:
        if len(s) == 1:
            return s[0]
        idx = p * (len(s) - 1)
        lo = int(idx)
        frac = idx - lo
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * frac

    return (q(0.25), q(0.50), q(0.75))


def leakage_audit(targets: Sequence[Target], as_of: int) -> None:
    """Raise if any label window opens at or before ``as_of``.

    Cheap, and run in tests over real sessions. A prediction that scores a
    window which already opened is not a prediction -- it is reading the answer.
    """
    offenders = [t for t in targets if t.entry_ts <= as_of]
    if offenders:
        sample = ", ".join(f"{t.symbol}@{t.entry_ts}" for t in offenders[:5])
        raise AssertionError(
            f"leakage: {len(offenders)} target window(s) open at or before as_of={as_of}: {sample}"
        )
    ranks = [t.universe_rank for t in targets]
    if ranks != list(range(1, len(ranks) + 1)):
        raise AssertionError("universe_rank must be dense and 1-based")
