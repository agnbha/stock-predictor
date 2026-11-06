"""Hard filters over the candidate inputs (design section 4.1).

A filter drops a symbol before ranking, so it changes which symbols can be
picked at all. That is a stronger claim than a feature makes, and every filter
is off by default; ``stock-predictor ablate`` measures what each would do.

A rule over a value that is not available -- not enough daily history yet, or
an opening check whose time has not come -- does not apply: the symbol is kept.
A filter must never remove a symbol for lack of data it could not have had.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, TypeVar

from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.features.candidates import candidate_values, is_missing, range_state_name
from stock_predictor.params import Params, get_params

T = TypeVar("T")

Rule = Callable[[dict, Params], Optional[bool]]


def _opening_check(v: dict, p: Params) -> Optional[bool]:
    state = range_state_name(v.get("cand_oc_range_state", float("nan")))
    above = v.get("cand_oc_above_vwap")
    if state is None or is_missing(above):
        return None
    ok = state in p.opening_check.allowed_range_states
    if p.opening_check.require_above_vwap:
        ok = ok and above >= 1.0
    return ok


def _between(col: str, lo: Callable[[Params], float], hi: Callable[[Params], float]) -> Rule:
    def rule(v: dict, p: Params) -> Optional[bool]:
        x = v.get(col)
        if is_missing(x):
            return None
        return lo(p) <= x <= hi(p)

    return rule


RULES: dict[str, Rule] = {
    "opening_check": _opening_check,
    "run_up": _between("cand_run_up_pct", lambda p: float("-inf"), lambda p: p.run_up.max_pct),
    "rsi": _between("cand_rsi", lambda p: p.rsi.min, lambda p: p.rsi.max),
    "volume": _between("cand_rel_volume", lambda p: p.volume.min_relative_volume, lambda p: float("inf")),
    "close_location": _between("cand_close_location", lambda p: p.close_location.min,
                               lambda p: p.close_location.max),
    "momentum": _between("cand_dist_from_high_pct", lambda p: -p.momentum.max_dist_from_high_pct,
                         lambda p: float("inf")),
}


def evaluate_rule(name: str, values: dict, params: Optional[Params] = None) -> Optional[bool]:
    """``True`` pass, ``False`` fail, ``None`` not applicable (value unavailable)."""
    return RULES[name](values, params or get_params())


def failed_rules(values: dict, names: Sequence[str], params: Optional[Params] = None) -> list[str]:
    params = params or get_params()
    return [n for n in names if evaluate_rule(n, values, params) is False]


def apply_filters(
    views: AnalyzerViews,
    session_date: str,
    as_of: int,
    ranked: Sequence[T],
    n: int,
    params: Optional[Params] = None,
    key: Callable[[T], str] = lambda x: x.symbol,  # type: ignore[attr-defined]
) -> list[T]:
    """Walk an already-ranked list, drop what the active filters reject, keep ``n``.

    With no active filter this is ``ranked[:n]`` and touches no data.
    """
    params = params or get_params()
    active = params.active_filters()
    if not active:
        return list(ranked)[:n]
    symbols = [key(r) for r in ranked]
    values = candidate_values(views, session_date, as_of, symbols, params)
    kept = [r for r in ranked if not failed_rules(values.get(key(r), {}), active, params)]
    return kept[:n]
