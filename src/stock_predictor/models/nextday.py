"""The next session: which stocks will trade X% above tomorrow's open before
its close, judged from where each stock stood at 15:00 today.

Before a session opens there are no candles for it, so the per-minute models
have nothing to work from. This model has one row per (symbol, session):

* **inputs**: the symbol's whole feature row (the same 84 columns) at 15:00
  on the session before, from the saved frames. That row was built only
  from candles up to then, and daily values from the sessions before that.
* **label**: did the session's high reach the open × (1 + X%)? Entry is the
  open, a price anyone can buy at through the pre-open auction; an overnight
  gap is not counted as a gain.
* **band**: the best gain available that session, the high against the open
  (p25 / p50 / p75), and the return to the close.

It is trained and evaluated exactly like the per-minute fixed targets: a
twin without the held-out month is checked on it, and the model used live
is trained on everything and calibrated on its last 20 sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from stock_predictor.clock import NSE
from stock_predictor.config import CONFIG, LAST_INTRADAY_POINT
from stock_predictor.eval.charges import round_trip_cost_pct
from stock_predictor.models.training import model_feature_columns

ENTRY = "next_open"  # ModelCard.entry for these models
OUTCOME_COLUMNS = ("open", "max_gain_pct", "close_return_pct", "label_class", "range_prev_pct")


# -- the calendar ---------------------------------------------------------------


def holidays(year: int, root: Optional[Path] = None) -> Optional[set[date]]:
    """NSE equity holidays for ``year`` from stock-analyzer's list beside its
    database, or None when it has no list for that year."""
    root = root or CONFIG.analyzer_db.parent.parent / "src" / "main" / "resources"
    path = Path(root) / f"nse-holidays-{year}.txt"
    if not path.exists():
        return None
    out = set()
    for line in path.read_text().splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            out.add(date.fromisoformat(token))
    return out


def next_trading_day(after: date, root: Optional[Path] = None) -> tuple[date, bool]:
    """The first weekday after ``after`` that is not an NSE holiday. The flag
    is False when a year had no holiday list, so the date is only a weekday."""
    d, known = after, True
    while True:
        d += timedelta(days=1)
        if d.weekday() >= 5:
            continue
        hs = holidays(d.year, root)
        if hs is None:
            known = False
        elif d in hs:
            continue
        return d, known


# -- rows: the state at 15:00, the next session's outcome --------------------


def state_point(session_date: str) -> int:
    return NSE.epoch_of(date.fromisoformat(session_date), LAST_INTRADAY_POINT)


def state_at(frame, as_of: int):
    """Each symbol's latest row at or before ``as_of``."""
    part = frame[frame["ts_epoch"] <= as_of]
    return part.sort_values("ts_epoch").groupby("symbol", as_index=False).tail(1).reset_index(drop=True)


def session_outcomes(views, session_date: str, target_pct: float):
    """Per symbol: the session's open, its best gain from the open, the return
    from the open to the close, and whether +target_pct was reached."""
    import pandas as pd

    rows = {}
    for c in views.intraday_candles(session_date):
        r = rows.get(c.symbol)
        if r is None:
            rows[c.symbol] = [c.open, c.high, c.low, c.close]
        else:
            r[1], r[2], r[3] = max(r[1], c.high), min(r[2], c.low), c.close
    out = []
    for sym, (op, hi, lo, cl) in rows.items():
        if not op or op <= 0:
            continue
        gain = max(hi / op - 1.0, 0.0) * 100.0
        out.append({"symbol": sym, "open": op, "max_gain_pct": gain,
                    "close_return_pct": (cl / op - 1.0) * 100.0,
                    "label_class": int(hi >= op * (1 + target_pct / 100.0)),
                    "range_pct": (hi - lo) / cl * 100.0 if cl else np.nan})
    return pd.DataFrame(out)


def next_session_set(views, source, sessions: Sequence[str], target_pct: float,
                     progress: Optional[Callable[[int, int, str], None]] = None):
    """One row per (symbol, session) for every session in ``sessions`` that has
    the session before it: features at 15:00 the session before, the outcome
    from this session's open. ``source`` is a FrameSource (saved frames)."""
    import pandas as pd

    parts = []
    prev_range = None
    for i, (prev, sd) in enumerate(zip(sessions, sessions[1:]), start=1):
        if progress:
            progress(i, len(sessions) - 1, sd)
        feats = source.features(prev)
        if feats.empty:
            prev_range = None
            continue
        state = state_at(feats, state_point(prev))
        out = session_outcomes(views, sd, target_pct)
        if out.empty:
            continue
        # The widest range the session before: a baseline, never a feature.
        if prev_range is None or prev_range[0] != prev:
            prev_range = (prev, session_outcomes(views, prev, target_pct)[["symbol", "range_pct"]])
        rows = state.merge(out.drop(columns=["range_pct"]), on="symbol", how="inner")
        rows = rows.merge(prev_range[1].rename(columns={"range_pct": "range_prev_pct"}), on="symbol", how="left")
        prev_range = (sd, out[["symbol", "range_pct"]])
        parts.append(rows.assign(session_date=sd, base_session=prev))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def feature_names(rows) -> list[str]:
    drop = set(OUTCOME_COLUMNS) | {"session_date", "base_session"}
    return [c for c in model_feature_columns(rows) if c not in drop]


def matrix(rows, names: Sequence[str]) -> np.ndarray:
    return np.asfortranarray(rows.reindex(columns=list(names)).to_numpy(dtype=np.float64))


# -- training ---------------------------------------------------------------------


def calibrate(rows, names: Sequence[str], last_n: int = 20):
    """Fit the classifier without the last ``last_n`` sessions and map its P
    on them to how often it came true. Returns (calibrator, table) or (None, [])."""
    from stock_predictor.eval.target import CALIBRATION_EDGES
    from stock_predictor.models.calibrate import GainCalibrator
    from stock_predictor.models.gbdt import GBDTClassifier

    sessions = sorted(rows["session_date"].unique())
    if len(sessions) < last_n + 20:
        return None, []
    held = set(sessions[-last_n:])
    early = ~rows["session_date"].isin(held)
    clf = GBDTClassifier().fit(matrix(rows[early], names), rows.loc[early, "label_class"].to_numpy())
    late = rows[~early]
    p = np.asarray(clf.predict_proba(matrix(late, names)))
    hit = late["label_class"].to_numpy(dtype=float)
    cal = GainCalibrator().fit(p, hit)
    table = []
    for lo, hi in zip(CALIBRATION_EDGES, CALIBRATION_EDGES[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p <= hi)
        if m.any():
            table.append({"band": f"{lo:.2f}-{hi:.2f}", "rows": int(m.sum()),
                          "predicted": float(p[m].mean()), "seen": float(hit[m].mean())})
    return cal, table


def fit(rows, names: Sequence[str], calibrate_last: int = 20):
    """A ForwardModel whose classifier predicts reaching the target from the
    open and whose band is the best gain from the open."""
    from stock_predictor.models.forward import ForwardModel

    model = ForwardModel(feature_names=list(names), band="max_gain_pct")
    model.fit_matrix(matrix(rows, names), rows["label_class"].to_numpy(),
                     rows["close_return_pct"].to_numpy(dtype=float), rows["max_gain_pct"].to_numpy(dtype=float))
    cal, table = calibrate(rows, names, calibrate_last) if calibrate_last > 0 else (None, [])
    model.calibrator = cal
    return model, table


# -- evaluation ------------------------------------------------------------------


@dataclass
class NextSessionReport:
    target_pct: float
    sessions: list[str]
    n: int
    cost_pct: float
    summary: dict
    calibration: list[dict]
    band_coverage: dict
    filters: list[dict]
    by_symbol: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "target": f"+{self.target_pct:g}% from the next session's open, before its close",
            "entry": ENTRY, "sessions": [self.sessions[0], self.sessions[-1]],
            "session_count": len(self.sessions), "top_n": self.n, "cost_pct": self.cost_pct,
            "summary": self.summary, "calibration": self.calibration,
            "band_coverage": self.band_coverage, "filters": self.filters, "by_symbol": self.by_symbol,
            # one "point": the open, so readers of per-point results find it
            "by_point": [{"point": "09:15", "base_hit": self.summary["base_hit"],
                          "model_hit": self.summary["model"]["hit"],
                          "intraday_volatility_hit": self.summary["intraday_volatility"]["hit"]}],
        }


def evaluate_next_session(model, rows, target_pct: float, n: int = 5, views=None) -> NextSessionReport:
    """Each session: the model's top ``n`` by P against every stock, the ``n``
    with the widest range the session before, and the ``n`` widest 20-day ATR.
    A trade buys at the open and sells at the target or the close."""
    from stock_predictor.eval.target import CALIBRATION_EDGES
    from stock_predictor.eval.baselines import highest_volatility

    rows = rows.copy()
    pred = model.predict(rows)
    rows["p"], rows["band_p25"], rows["band_p50"], rows["band_p75"] = pred.p_up, pred.gain_p25, pred.gain_p50, pred.gain_p75
    rows["p_seen"] = pred.p_seen if pred.p_seen is not None else np.nan
    cost = round_trip_cost_pct(1000.0, 1000.0 * (1 + target_pct / 100.0))
    rows["hit"] = rows["label_class"] > 0
    rows["net"] = np.where(rows["hit"], target_pct, rows["close_return_pct"]) - cost

    def top(col):
        return rows.sort_values(col, ascending=False, kind="stable").groupby("session_date").head(n)

    picks = {"model": top("p"), "intraday_volatility": top("range_prev_pct")}
    if views is not None:
        atr = {sd: {p.symbol for p in highest_volatility(views, sd, 0, n)} for sd in rows["session_date"].unique()}
        picks["highest_volatility"] = rows[[s in atr[sd] for s, sd in zip(rows["symbol"], rows["session_date"])]]
    summary = {"base_hit": float(rows["hit"].mean()), "base_net": float(rows["net"].mean())}
    for k, pk in picks.items():
        summary[k] = {"hit": float(pk["hit"].mean()), "net": float(pk["net"].mean())}
    summary["model"]["predicted"] = float(picks["model"]["p"].mean())
    summary["labels"] = {"intraday_volatility": f"the {n} with the widest range the session before",
                         "highest_volatility": f"the {n} widest 20-day ATR"}

    calibration = []
    for lo, hi in zip(CALIBRATION_EDGES, CALIBRATION_EDGES[1:]):
        m = ((rows["p"] > lo) & (rows["p"] <= hi)) if lo > 0 else (rows["p"] <= hi)
        if m.any():
            calibration.append({"band": f"({lo}, {hi}]", "rows": int(m.sum()), "predicted": float(rows.loc[m, "p"].mean()),
                                "hit": float(rows.loc[m, "hit"].mean()), "net": float(rows.loc[m, "net"].mean())})
    coverage = {"band": "max_gain_pct",
                **{q: float((rows["max_gain_pct"] <= rows[f"band_{q}"]).mean()) for q in ("p25", "p50", "p75")}}
    per_session = rows["session_date"].nunique()
    filters = []
    for rule, keep in (("chance >= 50% (model's P)", rows["p"] >= 0.5),
                       ("chance >= 50% (calibrated)", rows["p_seen"] >= 0.5),
                       (f"band median >= +{target_pct:g}%", rows["band_p50"] >= target_pct)):
        k = rows[keep.fillna(False).to_numpy(dtype=bool)]
        if len(k):
            filters.append({"rule": rule, "rows": int(len(k)), "per_session_point": len(k) / per_session,
                            "hit": float(k["hit"].mean()), "net": float(k["net"].mean())})
    from stock_predictor.eval.target import by_symbol_record

    sessions = sorted(rows["session_date"].unique())
    rows["chance"] = rows["p_seen"].fillna(rows["p"])
    return NextSessionReport(target_pct, sessions, n, cost, summary, calibration, coverage, filters,
                             by_symbol_record(rows))
