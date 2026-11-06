"""Assemble the stage-1 training matrix.

One row per (symbol, minute) over a set of sessions, with the full feature
frame and a triple-barrier label. Whole sessions go in or stay out -- the
walk-forward splitter works at day granularity (design section 6).

Two things keep this cheap as sessions accumulate:

* **Saved frames.** ``FrameSource`` reads each completed session's labelled
  frame from a ``FrameStore`` and builds (then saves) only what is missing or
  stale. ``FrameSource.at`` slices the same frame back to any earlier
  prediction point for evaluation, which is exact because every row depends
  only on candles at or before its own minute.
* **A compact matrix.** ``training_set`` keeps one float64 array of features
  plus the two label arrays, built a session at a time, rather than
  concatenating whole DataFrames and copying them into a matrix again.
  ``minute_step`` keeps every k-th minute: neighbouring minutes share most of
  a 30-minute label horizon, so they add rows faster than information.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Callable, Optional, Sequence

import numpy as np

from stock_predictor.clock import NSE
from stock_predictor.config import LAST_INTRADAY_POINT
from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.data.framestore import FrameStore, code_fingerprint, data_fingerprints, variant_key
from stock_predictor.features.pipeline import build_feature_frame
from stock_predictor.models.triple_barrier import TripleBarrierConfig, triple_barrier_labels
from stock_predictor.params import Params, get_params

# Columns that are identifiers, raw inputs, or labels -- never model features.
LABEL_COLUMNS = ("label_class", "forward_return_pct", "minutes_to_barrier")
NON_FEATURE_COLUMNS = {
    "symbol", "session_date", "as_of", "ts_epoch",
    "_open", "_high", "_low", "_close",
    *LABEL_COLUMNS,
}


def model_feature_columns(frame) -> list[str]:
    return [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]


def training_as_of(session_date: str) -> int:
    """Training frames run to the last prediction point; labels near it are
    cut off by the time barrier there."""
    return NSE.epoch_of(date.fromisoformat(session_date), LAST_INTRADAY_POINT)


def labelled_session_frame(
    views: AnalyzerViews,
    session_date: str,
    cfg: Optional[TripleBarrierConfig] = None,
    params: Optional[Params] = None,
):
    """One completed session's feature frame with triple-barrier labels."""
    import pandas as pd

    cfg = cfg or TripleBarrierConfig()
    # The whole session is visible for label construction; each row's
    # features were built from candles at or before that minute, and each
    # label looks forward from that minute.
    frame = build_feature_frame(views, session_date, training_as_of(session_date), params=params)
    if frame.empty:
        return frame
    frame = frame.sort_values(["symbol", "ts_epoch"]).reset_index(drop=True)
    labelled = []
    for _, g in frame.groupby("symbol"):
        g = g.reset_index(drop=True)
        atr_pct = (g["atr_14"] / g["_close"].replace(0, 1) * 100.0).tolist()
        outcomes = triple_barrier_labels(
            g["_close"].tolist(), g["_high"].tolist(), g["_low"].tolist(), atr_pct, cfg
        )
        g["label_class"] = [o.label for o in outcomes]
        g["forward_return_pct"] = [o.forward_return_pct for o in outcomes]
        g["minutes_to_barrier"] = [o.minutes_to_barrier for o in outcomes]
        labelled.append(g)
    return pd.concat(labelled, ignore_index=True) if labelled else pd.DataFrame()


def frame_variant(params: Params, cfg: TripleBarrierConfig) -> dict:
    """What a saved frame's columns and labels depend on: the candidate inputs
    switched on as features (filter switches and thresholds do not touch the
    frame) and the barrier settings."""
    cand = params.candidates()
    features = {}
    for name in params.active_features():
        sec = asdict(cand[name])
        sec.pop("filter", None)
        features[name] = sec
    return {"features": features, "triple_barrier": asdict(cfg)}


class FrameSource:
    """Labelled frames for completed sessions: from the store when it has a
    current one, otherwise built and saved. ``store=None`` always builds."""

    def __init__(
        self,
        views: AnalyzerViews,
        params: Optional[Params] = None,
        cfg: Optional[TripleBarrierConfig] = None,
        store: Optional[FrameStore] = None,
    ):
        self.views = views
        self.params = params or get_params()
        self.cfg = cfg or TripleBarrierConfig()
        self.store = store
        self.loaded = 0
        self.built = 0
        self._description = frame_variant(self.params, self.cfg)
        self._variant = variant_key(self._description)
        self._code: Optional[str] = None
        self._data: Optional[dict[str, str]] = None
        self._last = None  # (session, frame): evaluation asks for one session repeatedly

    def _key(self, session_date: str) -> Optional[str]:
        if self._code is None:
            self._code = code_fingerprint()
            self._data = data_fingerprints(self.views)
        data = self._data.get(session_date)
        if data is None:
            return None
        return variant_key({"code": self._code, "data": data})

    def session(self, session_date: str):
        """The full labelled frame for a completed session."""
        if self._last is not None and self._last[0] == session_date:
            return self._last[1]
        frame = self._session(session_date)
        self._last = (session_date, frame)
        return frame

    def _session(self, session_date: str):
        key = self._key(session_date) if self.store is not None else None
        if key is not None:
            frame = self.store.load(self._variant, session_date, key)
            if frame is not None:
                self.loaded += 1
                return frame
        frame = labelled_session_frame(self.views, session_date, self.cfg, self.params)
        self.built += 1
        if key is not None and not frame.empty:
            self.store.save(self._variant, self._description, session_date, key, frame)
        return frame

    def at(self, session_date: str, as_of: int):
        """The feature frame ``build_feature_frame`` would give at ``as_of``,
        cut from the saved session frame when that is exact, else built."""
        if self.store is None or as_of > training_as_of(session_date):
            return build_feature_frame(self.views, session_date, as_of, params=self.params)
        full = self.session(session_date)
        if full.empty:
            return full
        part = full[full["ts_epoch"] <= as_of]
        # A symbol with one candle so far is skipped by a build at as_of but
        # present in the full frame, where it also moves the cross-sectional
        # ranks of that minute. Rare; build those sessions directly.
        if part.empty or part.groupby("symbol")["ts_epoch"].size().min() < 2:
            return build_feature_frame(self.views, session_date, as_of, params=self.params)
        part = part.drop(columns=[c for c in LABEL_COLUMNS if c in part.columns])
        part = part.assign(as_of=as_of)
        return part.sort_values(["ts_epoch", "symbol"]).reset_index(drop=True)


def _on_step(frame, session_date: str, step: int):
    open_epoch = NSE.session_open_epoch(date.fromisoformat(session_date))
    return ((frame["ts_epoch"].to_numpy() - open_epoch) // 60) % step == 0


@dataclass
class TrainingSet:
    X: np.ndarray                    # rows x features, float64, column-major
    label_class: np.ndarray          # +1 / 0 / -1
    forward_return_pct: np.ndarray
    feature_names: list[str]
    sessions: list[str]
    minute_step: int = 1

    @property
    def rows(self) -> int:
        return int(self.X.shape[0])


def training_set(
    views: AnalyzerViews,
    sessions: Sequence[str],
    cfg: Optional[TripleBarrierConfig] = None,
    params: Optional[Params] = None,
    minute_step: int = 1,
    source: Optional[FrameSource] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> TrainingSet:
    """The stage-1 matrix over ``sessions``, holding one session's DataFrame at
    a time. ``minute_step`` = k keeps minutes 0, k, 2k, ... after the open."""
    if minute_step < 1:
        raise ValueError("minute_step must be >= 1")
    source = source or FrameSource(views, params, cfg)
    names: Optional[list[str]] = None
    xs, ys, rs, used = [], [], [], []
    for i, sd in enumerate(sessions, start=1):
        if progress:
            progress(i, len(sessions), sd)
        f = source.session(sd)
        if f.empty:
            continue
        if minute_step > 1:
            f = f[_on_step(f, sd, minute_step)]
        if names is None:
            names = model_feature_columns(f)
        xs.append(f.reindex(columns=names).to_numpy(dtype=np.float64))
        ys.append(f["label_class"].to_numpy(dtype=np.int8))
        rs.append(f["forward_return_pct"].to_numpy(dtype=np.float64))
        used.append(sd)
        del f
    if not xs:
        return TrainingSet(np.empty((0, 0)), np.empty(0, np.int8), np.empty(0), [], [], minute_step)
    # Column-major, as the GBDT backends bin feature by feature: scikit-learn
    # copies a row-major matrix into this layout before fitting, which would
    # cost one more full-size copy at the peak.
    X = np.empty((sum(len(x) for x in xs), len(names or [])), dtype=np.float64, order="F")
    np.concatenate(xs, axis=0, out=X)
    del xs
    return TrainingSet(X, np.concatenate(ys), np.concatenate(rs), names or [], used, minute_step)


def training_frame(
    views: AnalyzerViews,
    sessions: Sequence[str],
    cfg: Optional[TripleBarrierConfig] = None,
    params: Optional[Params] = None,
    source: Optional[FrameSource] = None,
):
    """The labelled frames for ``sessions`` as one DataFrame. For inspection;
    training uses ``training_set``, which needs far less memory."""
    import pandas as pd

    source = source or FrameSource(views, params, cfg)
    parts = [f for f in (source.session(sd) for sd in sessions) if not f.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
