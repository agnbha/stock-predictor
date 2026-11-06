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
from stock_predictor.filters import apply_filters
from stock_predictor.labels.targets import adverse_excursions, build_targets, leakage_audit
from stock_predictor.params import Params, get_params

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
    params: Optional[Params] = None,
) -> dict:
    k = k or CONFIG.top_n
    params = params or get_params()
    per_session = []
    for sd in sessions:
        as_of = as_of_for(sd, clock_time)
        truth = build_targets(views, sd, as_of, top_n=k)
        if audit:
            leakage_audit(truth, as_of)
        if not truth:
            continue
        picks = pick_fn(views, sd, as_of)
        adverse = adverse_excursions(views, sd, as_of)
        per_session.append(
            M.evaluate(picks, truth, k, adverse=adverse,
                       adverse_move_pct=params.score_order.adverse_move_pct)
        )
    summary = M.aggregate(per_session)
    summary["sessions_scored"] = float(len(per_session))
    return summary


def backtest_baselines(
    views: AnalyzerViews,
    sessions: Sequence[str],
    names: Optional[Sequence[str]] = None,
    clock_time=OPENING_RANGE_T1,
    k: Optional[int] = None,
    params: Optional[Params] = None,
) -> dict[str, dict]:
    """Every baseline over ``sessions``, with any active filters applied to its
    ranking before the top ``k`` are taken."""
    names = names or list(BASELINES)
    params = params or get_params()
    n = k or CONFIG.top_n
    out: dict[str, dict] = {}
    for name in names:
        def pick(v, sd, ao, _name=name):
            ranked = run_baseline(_name, v, sd, ao, 10**6)  # the full ordering
            return apply_filters(v, sd, ao, ranked, n, params)

        out[name] = backtest_picks(views, sessions, pick, clock_time, k, params=params)
    return out
