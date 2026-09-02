"""The thin calibration that sits on top of stage 2 (design section 5).

Maps stage-1 scores onto realistic gain expectations -- a handful of
parameters. This is the one place the 66-session limit genuinely binds:
fitting a few parameters on 66 days is defensible where fitting a full ranker
on them would not be. Kept to an isotonic (monotone, non-parametric) fit, or a
1-parameter linear scale when SciPy/sklearn are absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class GainCalibrator:
    _x: list[float] = field(default_factory=list)
    _y: list[float] = field(default_factory=list)
    _scale: float = 1.0
    _iso = None
    fitted: bool = False

    def fit(self, scores: Sequence[float], realised_gain_pct: Sequence[float]) -> "GainCalibrator":
        xs = list(map(float, scores))
        ys = list(map(float, realised_gain_pct))
        if len(xs) < 8:
            # too little to shape a curve; fall back to a single scale factor
            num = sum(x * y for x, y in zip(xs, ys))
            den = sum(x * x for x in xs) or 1.0
            self._scale = num / den
            self.fitted = True
            return self
        try:
            from sklearn.isotonic import IsotonicRegression

            self._iso = IsotonicRegression(out_of_bounds="clip").fit(xs, ys)
        except Exception:
            order = sorted(zip(xs, ys))
            self._x = [x for x, _ in order]
            self._y = [y for _, y in order]
        self.fitted = True
        return self

    def transform(self, score: float) -> float:
        if not self.fitted:
            return score
        if self._iso is not None:
            return float(self._iso.predict([score])[0])
        if self._x:
            return _interp(score, self._x, self._y)
        return score * self._scale

    def __call__(self, score: float) -> float:
        return self.transform(score)


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo] or 1.0
    return ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / span
