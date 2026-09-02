"""Run baselines (and, when given, a model's picks) across sessions and
summarise. Phase 2 of delivery: the harness that makes every later number mean
something.
"""

from __future__ import annotations

from datetime import date
from typing import Callable, Optional, Sequence

from stock_predictor.clock import NSE
from stock_predictor.config import CONFIG, OPENING_RANGE_T1
from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.eval import metrics as M
from stock_predictor.eval.baselines import BASELINES, run_baseline
from stock_predictor.labels.targets import build_targets, leakage_audit

PickFn = Callable[[AnalyzerViews, str, int], "list[M.Pick]"]


def as_of_for(session_date: str, clock_time=OPENING_RANGE_T1) -> int:
    return NSE.epoch_of(date.fromisoformat(session_date), clock_time)


def backtest_picks(
    views: AnalyzerViews,
    sessions: Sequence[str],
    pick_fn: PickFn,
    clock_time=OPENING_RANGE_T1,
    k: Optional[int] = None,
    audit: bool = True,
) -> dict:
    k = k or CONFIG.top_n
    per_session = []
    for sd in sessions:
        as_of = as_of_for(sd, clock_time)
        truth = build_targets(views, sd, as_of, top_n=k)
        if audit:
            leakage_audit(truth, as_of)
        if not truth:
            continue
        picks = pick_fn(views, sd, as_of)
        per_session.append(M.evaluate(picks, truth, k))
    summary = M.aggregate(per_session)
    summary["sessions_scored"] = float(len(per_session))
    return summary


def backtest_baselines(
    views: AnalyzerViews,
    sessions: Sequence[str],
    names: Optional[Sequence[str]] = None,
    clock_time=OPENING_RANGE_T1,
    k: Optional[int] = None,
) -> dict[str, dict]:
    names = names or list(BASELINES)
    out: dict[str, dict] = {}
    for name in names:
        out[name] = backtest_picks(
            views, sessions,
            lambda v, sd, ao, _n=name: run_baseline(_n, v, sd, ao, k or CONFIG.top_n),
            clock_time, k,
        )
    return out
