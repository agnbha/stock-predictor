"""Stage 1 -- the per-candle model, trained where the data is (design section 5).

One row per (symbol, minute). Two heads share the feature matrix:

* a classifier for P(upper barrier hit first over the next H minutes);
* a quantile regressor for the forward-return band (p25 / p50 / p75).

This stage sees over a million examples and is trained without ever knowing
what a "top five" is. Stage 2 reduces its output into the daily answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from stock_predictor.models.calibrate import GainCalibrator
from stock_predictor.models.gbdt import GBDTClassifier, GBDTQuantileRegressor, backend_name


@dataclass
class ForwardPrediction:
    p_up: np.ndarray            # P(upper barrier first)
    gain_p25: np.ndarray
    gain_p50: np.ndarray
    gain_p75: np.ndarray
    # p_up mapped to how often that probability came true on sessions the
    # model had not seen; None when the model carries no calibrator.
    p_seen: Optional[np.ndarray] = None

    @property
    def expected_move(self) -> np.ndarray:
        """Predicted move discounted by its own probability -- the quantity
        stage 2 ranks on."""
        return np.maximum(self.gain_p50, 0.0) * self.p_up


@dataclass
class ForwardModel:
    feature_names: Sequence[str]
    _clf: GBDTClassifier = field(default_factory=GBDTClassifier)
    _qr: GBDTQuantileRegressor = field(default_factory=GBDTQuantileRegressor)
    trained: bool = False
    calibrator: Optional[GainCalibrator] = None  # P(up) -> rate seen (train --target)

    @property
    def backend(self) -> str:
        return backend_name()

    def _matrix(self, frame) -> np.ndarray:
        return np.asarray(frame[list(self.feature_names)].to_numpy(dtype=float))

    def fit(self, frame, label_class: Sequence[int], forward_return_pct: Sequence[float]) -> "ForwardModel":
        return self.fit_matrix(self._matrix(frame), label_class, forward_return_pct)

    def fit_matrix(self, X: np.ndarray, label_class: Sequence[int],
                   forward_return_pct: Sequence[float]) -> "ForwardModel":
        """Fit on a ready matrix whose columns are ``feature_names``, in order
        (``training_set`` builds one)."""
        y_up = (np.asarray(label_class) > 0).astype(int)
        self._clf.fit(X, y_up)
        self._qr.fit(X, np.asarray(forward_return_pct, dtype=float))
        self.trained = True
        return self

    def predict(self, frame) -> ForwardPrediction:
        if not self.trained:
            raise RuntimeError("ForwardModel.predict called before fit")
        X = self._matrix(frame)
        q = self._qr.predict(X)
        p = np.asarray(self._clf.predict_proba(X))
        cal = getattr(self, "calibrator", None)
        return ForwardPrediction(
            p_up=p,
            gain_p25=np.asarray(q[0.25]),
            gain_p50=np.asarray(q[0.5]),
            gain_p75=np.asarray(q[0.75]),
            p_seen=None if cal is None else np.array([cal.transform(x) for x in p]),
        )
