"""How often a fixed-target model's top picks reach the target.

For a model trained on a fixed target (``train --target 1 --horizon 40``): at
every prediction point of every session, take each symbol's latest row, rank
the symbols by the model's P(target), and look at what the top ``n`` then did:

* **hit**: touched the target within the horizon (before the stop, if any);
* **stopped**: reached the stop first (only with a stop);
* **net**: a trade bought at that close and sold at the target, the stop or
  the end of the window, less round-trip costs.

The same for two rankings that need no model, and for every symbol (the base
rate). A fixed target is mechanically easier for a volatile stock, so the
first of those, the most volatile stocks right now, is the one to beat.
Rows whose window runs past the close are left out, as in training.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import Optional, Sequence

import numpy as np

from stock_predictor.clock import NSE
from stock_predictor.eval.baselines import highest_volatility
from stock_predictor.eval.charges import round_trip_cost_pct
from stock_predictor.models.triple_barrier import TripleBarrierConfig

STRATEGIES = ("model", "intraday_volatility", "highest_volatility")
CALIBRATION_EDGES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0)


def default_points(cfg: TripleBarrierConfig, step_minutes: int = 15) -> list[time]:
    """09:30, then every ``step_minutes`` while the window still ends inside
    the session (to 15:00 for a window that runs to the close)."""
    last = 15 * 60 if cfg.to_close else 15 * 60 + 29 - cfg.horizon_minutes  # minutes after midnight
    out = []
    m = 9 * 60 + 30
    while m <= last:
        out.append(time(m // 60, m % 60))
        m += step_minutes
    return out


def trade_net_pct(label, forward_return_pct, cfg: TripleBarrierConfig, cost_pct: float) -> np.ndarray:
    """Buy at the row's close; sell at the target, the stop, or the close at
    the end of the window. Net of ``cost_pct``."""
    label = np.asarray(label)
    out = np.where(label > 0, cfg.target_pct, np.asarray(forward_return_pct, dtype=float))
    if cfg.stop_pct is not None:
        out = np.where(label < 0, -cfg.stop_pct, out)
    return out - cost_pct


@dataclass
class TargetReport:
    cfg: TripleBarrierConfig
    sessions: list[str]
    n: int
    cost_pct: float
    by_point: "object"      # DataFrame: one row per point
    calibration: "object"   # DataFrame: predicted vs realised, by probability band
    summary: dict
    band_coverage: Optional[dict] = None  # share of outcomes at or below the predicted p25/p50/p75
    filters: list = ()                    # "keep the symbols that ..." rules and how they did

    def to_dict(self) -> dict:
        return {
            "target": self.cfg.describe(), "sessions": [self.sessions[0], self.sessions[-1]],
            "session_count": len(self.sessions), "top_n": self.n, "cost_pct": self.cost_pct,
            "summary": self.summary, "band_coverage": self.band_coverage, "filters": list(self.filters),
        }


def point_rows(source, sessions: Sequence[str], points: Sequence[time]):
    """Each symbol's latest row at every point, labelled, from ``source``
    (a FrameSource whose ``cfg`` is the target)."""
    import pandas as pd

    parts = []
    for sd in sessions:
        f = source.session(sd)
        if f.empty:
            continue
        day = date.fromisoformat(sd)
        f = f.sort_values("ts_epoch")
        for p in points:
            as_of = NSE.epoch_of(day, p)
            r = f[f["ts_epoch"] <= as_of].groupby("symbol", as_index=False).tail(1)
            r = r[r["label_complete"].to_numpy(dtype=bool)]
            if not r.empty:
                parts.append(r.assign(session_date=sd, point=p.strftime("%H:%M")))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def evaluate_target(
    model,
    source,
    sessions: Sequence[str],
    n: int = 5,
    points: Optional[Sequence[time]] = None,
) -> TargetReport:
    import pandas as pd

    cfg = source.cfg
    if not cfg.fixed:
        raise ValueError("evaluate_target needs a model trained on a fixed target (--target)")
    points = list(points or default_points(cfg))
    rows = point_rows(source, sessions, points)
    if rows.empty:
        raise ValueError("no rows to evaluate")
    cost = round_trip_cost_pct(1000.0, 1000.0 * (1 + cfg.target_pct / 100.0))
    pred = model.predict(rows)
    rows["p"] = pred.p_up
    rows["p_seen"] = pred.p_seen if pred.p_seen is not None else np.nan
    rows["band_p25"], rows["band_p50"], rows["band_p75"] = pred.gain_p25, pred.gain_p50, pred.gain_p75
    rows["intraday_volatility"] = rows["atr_14"] / rows["_close"].replace(0, np.nan)
    rows["hit"] = rows["label_class"] > 0
    rows["stopped"] = rows["label_class"] < 0
    rows["net"] = trade_net_pct(rows["label_class"], rows["forward_return_pct"], cfg, cost)
    used = sorted(rows["session_date"].unique())
    calm = {sd: {p.symbol for p in highest_volatility(source.views, sd, 0, n)} for sd in used}

    def top(g, col):
        return g.sort_values(col, ascending=False, kind="stable").groupby("session_date").head(n)

    table = []
    for point, g in rows.groupby("point"):
        picks = {
            "model": top(g, "p"),
            "intraday_volatility": top(g, "intraday_volatility"),
            "highest_volatility": g[[s in calm[sd] for s, sd in zip(g["symbol"], g["session_date"])]],
        }
        rec = {"point": point, "base_hit": g["hit"].mean(), "base_net": g["net"].mean()}
        for name, pk in picks.items():
            rec[f"{name}_hit"] = pk["hit"].mean()
            rec[f"{name}_stopped"] = pk["stopped"].mean()
            rec[f"{name}_net"] = pk["net"].mean()
        rec["model_p"] = picks["model"]["p"].mean()
        table.append(rec)
    by_point = pd.DataFrame(table)

    rows["band"] = pd.cut(rows["p"], list(CALIBRATION_EDGES), include_lowest=True)
    calibration = rows.groupby("band", observed=True).agg(
        rows=("p", "size"), predicted=("p", "mean"), hit=("hit", "mean"), net=("net", "mean"),
    ).reset_index()

    summary = {"base_hit": float(by_point["base_hit"].mean()), "base_net": float(by_point["base_net"].mean())}
    for name in STRATEGIES:
        summary[name] = {
            "hit": float(by_point[f"{name}_hit"].mean()),
            "net": float(by_point[f"{name}_net"].mean()),
        }
        if cfg.stop_pct is not None:
            summary[name]["stopped"] = float(by_point[f"{name}_stopped"].mean())
    summary["model"]["predicted"] = float(by_point["model_p"].mean())

    band_name = getattr(model, "band", "forward_return_pct")
    actual = rows[band_name] if band_name in rows else rows["forward_return_pct"]
    coverage = {"band": band_name,
                **{q: float((actual <= rows[f"band_{q}"]).mean()) for q in ("p25", "p50", "p75")}}

    points_seen = rows.groupby(["session_date", "point"]).ngroups
    filters = []
    for rule, keep in (
        (f"chance >= 50% (model's P)", rows["p"] >= 0.5),
        (f"chance >= 50% (calibrated)", rows["p_seen"] >= 0.5),
        (f"band median >= +{cfg.target_pct:g}%", rows["band_p50"] >= cfg.target_pct),
    ):
        k = rows[keep.fillna(False).to_numpy(dtype=bool)]
        if len(k):
            filters.append({"rule": rule, "rows": int(len(k)), "per_session_point": len(k) / points_seen,
                            "hit": float(k["hit"].mean()), "net": float(k["net"].mean())})
    return TargetReport(cfg, used, n, cost, by_point, calibration, summary, coverage, filters)
