"""Assemble the stage-1 training matrix.

One row per (symbol, minute) over a set of sessions, with the full feature
frame and a triple-barrier label. Whole sessions go in or stay out -- the
walk-forward splitter works at day granularity (design section 6).

Labels look forward from each row over the session's full minute candles, to
the last minute, though the frame's rows stop at the last prediction point.
A row whose window runs past the close has an outcome cut short; it is kept in
the frame (``label_complete`` False) but not trained on.

Two things keep this cheap as sessions accumulate:

* **Saved frames.** ``FrameSource`` reads each completed session's feature
  frame from a ``FrameStore`` and builds (then saves) only what is missing or
  stale, then attaches labels for its target. Labels take well under a second
  a session, so a new target or horizon reuses every saved frame.
  ``FrameSource.at`` slices the same frame back to any earlier
  prediction point for evaluation, which is exact because every row depends
  only on candles at or before its own minute.
* **A compact matrix.** ``training_set`` keeps one float64 array of features
  plus the two label arrays, built a session at a time, rather than
  concatenating whole DataFrames and copying them into a matrix again.
  ``minute_step`` keeps every k-th minute: neighbouring minutes share most of
  a 30-minute label horizon, so they add rows faster than information.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from stock_predictor.clock import NSE
from stock_predictor.config import LAST_INTRADAY_POINT
from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.data.framestore import FrameStore, code_fingerprint, data_fingerprints, variant_key
from stock_predictor.features.pipeline import build_feature_frame
from stock_predictor.features.session import session_features, training_as_of  # noqa: F401 (re-exported)
from stock_predictor.models.triple_barrier import TripleBarrierConfig, barrier_arrays
from stock_predictor.params import Params, get_params

# Columns that are identifiers, raw inputs, or labels -- never model features.
LABEL_COLUMNS = ("label_class", "forward_return_pct", "max_gain_pct", "minutes_to_barrier",
                 "label_complete")
NON_FEATURE_COLUMNS = {
    "symbol", "session_date", "as_of", "ts_epoch",
    "_open", "_high", "_low", "_close",
    *LABEL_COLUMNS,
}


def model_feature_columns(frame) -> list[str]:
    return [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]


def attach_labels(views: AnalyzerViews, session_date: str, frame, cfg: Optional[TripleBarrierConfig] = None):
    """``frame`` with label columns for ``cfg``, each row's window read from the
    session's full minute candles (sorted by symbol, then time)."""
    cfg = cfg or TripleBarrierConfig()
    if frame.empty:
        return frame
    frame = frame.drop(columns=[c for c in LABEL_COLUMNS if c in frame.columns])
    frame = frame.sort_values(["symbol", "ts_epoch"]).reset_index(drop=True)
    last_ts = NSE.session_close_epoch(date.fromisoformat(session_date)) - 60
    candles: dict[str, list] = {}
    for c in views.intraday_candles(session_date):
        candles.setdefault(c.symbol, []).append(c)

    n = len(frame)
    out = {"label_class": np.zeros(n, np.int8), "forward_return_pct": np.zeros(n),
           "max_gain_pct": np.zeros(n), "minutes_to_barrier": np.zeros(n, np.int64),
           "label_complete": np.zeros(n, bool)}
    row_ts = frame["ts_epoch"].to_numpy()
    row_atr = (frame["atr_14"] / frame["_close"].replace(0, 1) * 100.0).to_numpy()
    for sym, rows in frame.groupby("symbol", sort=False).indices.items():
        cs = candles.get(sym, [])
        ts = np.array([c.ts_epoch for c in cs], dtype=np.int64)
        pos = np.searchsorted(ts, row_ts[rows])
        if len(ts) == 0 or (pos >= len(ts)).any() or (ts[np.minimum(pos, len(ts) - 1)] != row_ts[rows]).any():
            raise ValueError(f"{session_date} {sym}: frame rows without a matching candle")
        atr = np.zeros(len(ts))
        atr[pos] = row_atr[rows]
        a = barrier_arrays([c.close for c in cs], [c.high for c in cs], [c.low for c in cs],
                           atr, cfg, ts=ts, last_ts=last_ts)
        out["label_class"][rows] = a["label"][pos]
        out["forward_return_pct"][rows] = a["forward_return_pct"][pos]
        out["max_gain_pct"][rows] = a["max_gain_pct"][pos]
        out["minutes_to_barrier"][rows] = a["minutes_to_barrier"][pos]
        out["label_complete"][rows] = a["complete"][pos]
    for k, v in out.items():
        frame[k] = v
    return frame


def labelled_session_frame(
    views: AnalyzerViews,
    session_date: str,
    cfg: Optional[TripleBarrierConfig] = None,
    params: Optional[Params] = None,
):
    """One completed session's feature frame with labels for ``cfg``."""
    return attach_labels(views, session_date, session_features(views, session_date, params), cfg)


def frame_variant(params: Params) -> dict:
    """What a saved frame's columns depend on: the candidate inputs switched
    on as features (filter switches and thresholds do not touch the frame).
    Labels are not saved, so the target is not part of it."""
    cand = params.candidates()
    features = {}
    for name in params.active_features():
        sec = asdict(cand[name])
        sec.pop("filter", None)
        features[name] = sec
    return {"features": features}


def _build_frames(job) -> int:
    """Worker for ``FrameSource.prefetch``: build and save some sessions."""
    db_path, params, root, sessions = job
    with AnalyzerViews(Path(db_path)) as views:
        source = FrameSource(views, params, store=FrameStore(Path(root)))
        for sd in sessions:
            source.features(sd)
        return source.built


class FrameSource:
    """Labelled frames for completed sessions: features from the store when it
    has a current frame, otherwise built and saved; labels for ``cfg``
    attached on top. ``store=None`` always builds."""

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
        self._description = frame_variant(self.params)
        self._variant = variant_key(self._description)
        self._code: Optional[str] = None
        self._data: Optional[dict[str, str]] = None
        # evaluation asks for one session repeatedly
        self._last_features = None  # (session, frame)
        self._last = None           # (session, labelled frame)

    def _key(self, session_date: str) -> Optional[str]:
        if self._code is None:
            self._code = code_fingerprint()
            self._data = data_fingerprints(self.views)
        data = self._data.get(session_date)
        if data is None:
            return None
        return variant_key({"code": self._code, "data": data})

    def features(self, session_date: str):
        """The saved (or freshly built) feature frame for a completed session."""
        if self._last_features is not None and self._last_features[0] == session_date:
            return self._last_features[1]
        frame = self._features(session_date)
        self._last_features = (session_date, frame)
        return frame

    def _features(self, session_date: str):
        key = self._key(session_date) if self.store is not None else None
        if key is not None:
            frame = self.store.load(self._variant, session_date, key)
            if frame is not None:
                self.loaded += 1
                return frame
        frame = session_features(self.views, session_date, self.params)
        self.built += 1
        if key is not None and not frame.empty:
            self.store.save(self._variant, self._description, session_date, key, frame)
        return frame

    def session(self, session_date: str):
        """The full labelled frame for a completed session."""
        if self._last is not None and self._last[0] == session_date:
            return self._last[1]
        frame = attach_labels(self.views, session_date, self.features(session_date), self.cfg)
        self._last = (session_date, frame)
        return frame

    def prefetch(self, sessions: Sequence[str], jobs: int) -> int:
        """Build the saved frames ``sessions`` are missing in ``jobs`` processes;
        returns how many were built. Each one takes ~7 s, so the first run
        over months of sessions is where this pays."""
        if self.store is None or jobs <= 1:
            return 0
        missing = [sd for sd in sessions
                   if (k := self._key(sd)) is not None and not self.store.has(self._variant, sd, k)]
        if len(missing) < 2:
            return 0
        from concurrent.futures import ProcessPoolExecutor

        jobs = min(jobs, len(missing))
        work = [(str(self.views.db_path), self.params, str(self.store.root), missing[i::jobs])
                for i in range(jobs)]
        with ProcessPoolExecutor(jobs) as ex:
            built = sum(ex.map(_build_frames, work))
        self.built += built
        return built

    def at(self, session_date: str, as_of: int):
        """The feature frame ``build_feature_frame`` would give at ``as_of``,
        cut from the saved session frame when that is exact, else built."""
        if self.store is None or as_of > training_as_of(session_date):
            return build_feature_frame(self.views, session_date, as_of, params=self.params)
        full = self.features(session_date)
        if full.empty:
            return full
        part = full[full["ts_epoch"] <= as_of]
        # A symbol with one candle so far is skipped by a build at as_of but
        # present in the full frame, where it also moves the cross-sectional
        # ranks of that minute. Rare; build those sessions directly.
        if part.empty or part.groupby("symbol")["ts_epoch"].size().min() < 2:
            return build_feature_frame(self.views, session_date, as_of, params=self.params)
        part = part.assign(as_of=as_of)
        return part.sort_values(["ts_epoch", "symbol"]).reset_index(drop=True)


def default_jobs() -> int:
    """Processes for building frames: half the cores, at most 8 (each holds a
    session's candles and frame, a few hundred MB)."""
    return max(1, min(8, (os.cpu_count() or 2) // 2))


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
    row_session: Optional[np.ndarray] = None  # index into ``sessions`` per row
    max_gain_pct: Optional[np.ndarray] = None  # best high in each row's window, vs its close

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
    a time. Rows whose label window runs past the close are left out.
    ``minute_step`` = k keeps minutes 0, k, 2k, ... after the open."""
    if minute_step < 1:
        raise ValueError("minute_step must be >= 1")
    source = source or FrameSource(views, params, cfg)
    names: Optional[list[str]] = None
    xs, ys, rs, gs, used, idx = [], [], [], [], [], []
    for i, sd in enumerate(sessions, start=1):
        if progress:
            progress(i, len(sessions), sd)
        f = source.session(sd)
        if f.empty:
            continue
        keep = f["label_complete"].to_numpy(dtype=bool)
        if minute_step > 1:
            keep &= _on_step(f, sd, minute_step)
        f = f[keep]
        if names is None:
            names = model_feature_columns(f)
        xs.append(f.reindex(columns=names).to_numpy(dtype=np.float64))
        ys.append(f["label_class"].to_numpy(dtype=np.int8))
        rs.append(f["forward_return_pct"].to_numpy(dtype=np.float64))
        gs.append(f["max_gain_pct"].to_numpy(dtype=np.float64))
        idx.append(np.full(len(f), len(used), dtype=np.int32))
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
    return TrainingSet(X, np.concatenate(ys), np.concatenate(rs), names or [], used, minute_step,
                       np.concatenate(idx), np.concatenate(gs))


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
