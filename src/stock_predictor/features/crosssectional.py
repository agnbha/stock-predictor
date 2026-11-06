"""Cross-sectional features -- the group that makes a ranker work (design
section 4).

Every per-symbol feature also enters as its **rank and z-score within that
day's universe**, at the same minute. The question is "which of these 49 is
best today", so relative standing matters more than absolute value, and
day-level normalisation strips out the market-wide move.
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Iterable, Sequence

# Columns that are meaningful to rank cross-sectionally (skip the ones that are
# already relative, cyclical, or constant within a day).
_SKIP = {
    "symbol", "session_date", "as_of", "ts_epoch",
    "_open", "_high", "_low", "_close",
    "minutes_since_open", "session_progress_sin", "session_progress_cos", "day_of_week",
    "event_opening_range_breakout", "event_vwap_reclaim", "event_volume_spike",
    "event_prior_day_high_break", "gap_pct", "trend_score",
    # categorical states, not magnitudes
    "cand_oc_range_state", "cand_oc_above_vwap",
}


def cross_sectional_columns(base_columns: Iterable[str]) -> list[str]:
    out = []
    for c in base_columns:
        if c in _SKIP:
            continue
        out.append(f"{c}__xs_rank")
        out.append(f"{c}__xs_z")
    return out


def add_cross_sectional(rows: list[dict]) -> list[dict]:
    """Add ``<feat>__xs_rank`` (0..1) and ``<feat>__xs_z`` for each feature,
    computed within each ``ts_epoch`` group. Mutates and returns ``rows``."""
    if not rows:
        return rows
    feats = [c for c in rows[0] if c not in _SKIP]

    by_minute: dict[int, list[dict]] = {}
    for r in rows:
        by_minute.setdefault(r["ts_epoch"], []).append(r)

    for group in by_minute.values():
        for f in feats:
            # NaN means "not available" (e.g. too little daily history): it is
            # ranked against nothing and stays NaN, rather than posing as zero.
            vals = [(i, _num(g.get(f))) for i, g in enumerate(group)]
            present = [(i, v) for i, v in vals if v == v]
            n = len(present)
            order = sorted(present, key=lambda kv: kv[1])
            rank_of = {idx: pos for pos, (idx, _) in enumerate(order)}
            col = [v for _, v in present]
            mu = mean(col) if col else 0.0
            sd = pstdev(col) if n > 1 else 0.0
            for i, g in enumerate(group):
                if i not in rank_of:
                    g[f"{f}__xs_rank"] = _NAN
                    g[f"{f}__xs_z"] = _NAN
                    continue
                g[f"{f}__xs_rank"] = rank_of[i] / (n - 1) if n > 1 else 0.5
                g[f"{f}__xs_z"] = (vals[i][1] - mu) / sd if sd > 0 else 0.0
    return rows


_NAN = float("nan")


def _num(x) -> float:
    """None -> 0.0 (the historic default for absent values); NaN stays NaN."""
    return 0.0 if x is None else float(x)
