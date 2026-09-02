"""Gradient-boosted trees, with a portable backend.

Design section 5: "Gradient-boosted trees throughout ... nothing here calls for
a neural network", and it names LightGBM. LightGBM needs an OpenMP runtime that
is not always present, so this module prefers it and falls back to
scikit-learn's histogram GBDT (``HistGradientBoosting*``), which is the same
family of model. ``backend_name()`` reports which is active.
"""

from __future__ import annotations

from typing import Optional, Sequence

_BACKEND = None


def backend_name() -> str:
    _select()
    return _BACKEND  # type: ignore[return-value]


def _select() -> str:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        import lightgbm  # noqa: F401

        # importing is not enough -- the native lib must actually load
        lightgbm.Booster  # noqa: B018
        _BACKEND = "lightgbm"
    except Exception:
        try:
            import sklearn  # noqa: F401

            _BACKEND = "sklearn"
        except Exception:
            _BACKEND = "none"
    return _BACKEND


class GBDTClassifier:
    """Binary classifier: P(upper barrier hit first)."""

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 4,
                 random_state: int = 0):
        self.params = dict(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            random_state=random_state,
        )
        self._model = None
        self._backend = _select()

    def fit(self, X, y) -> "GBDTClassifier":
        if self._backend == "lightgbm":
            import lightgbm as lgb

            self._model = lgb.LGBMClassifier(
                n_estimators=self.params["n_estimators"],
                learning_rate=self.params["learning_rate"],
                max_depth=self.params["max_depth"],
                num_leaves=2 ** self.params["max_depth"],
                random_state=self.params["random_state"],
                verbose=-1,
            ).fit(X, y)
        elif self._backend == "sklearn":
            from sklearn.ensemble import HistGradientBoostingClassifier

            self._model = HistGradientBoostingClassifier(
                max_iter=self.params["n_estimators"],
                learning_rate=self.params["learning_rate"],
                max_depth=self.params["max_depth"],
                random_state=self.params["random_state"],
            ).fit(X, y)
        else:
            raise RuntimeError("no GBDT backend available; install stock-predictor[models]")
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)[:, 1]


class GBDTQuantileRegressor:
    """One regressor per quantile -- the p25 / p50 / p75 gain band."""

    def __init__(self, quantiles: Sequence[float] = (0.25, 0.5, 0.75), n_estimators: int = 300,
                 learning_rate: float = 0.05, max_depth: int = 4, random_state: int = 0):
        self.quantiles = tuple(quantiles)
        self.params = dict(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            random_state=random_state,
        )
        self._models: dict[float, object] = {}
        self._backend = _select()

    def fit(self, X, y) -> "GBDTQuantileRegressor":
        for q in self.quantiles:
            if self._backend == "lightgbm":
                import lightgbm as lgb

                m = lgb.LGBMRegressor(
                    objective="quantile", alpha=q,
                    n_estimators=self.params["n_estimators"],
                    learning_rate=self.params["learning_rate"],
                    max_depth=self.params["max_depth"],
                    num_leaves=2 ** self.params["max_depth"],
                    random_state=self.params["random_state"], verbose=-1,
                ).fit(X, y)
            elif self._backend == "sklearn":
                from sklearn.ensemble import HistGradientBoostingRegressor

                m = HistGradientBoostingRegressor(
                    loss="quantile", quantile=q,
                    max_iter=self.params["n_estimators"],
                    learning_rate=self.params["learning_rate"],
                    max_depth=self.params["max_depth"],
                    random_state=self.params["random_state"],
                ).fit(X, y)
            else:
                raise RuntimeError("no GBDT backend available; install stock-predictor[models]")
            self._models[q] = m
        return self

    def predict(self, X) -> dict[float, "Sequence[float]"]:
        return {q: m.predict(X) for q, m in self._models.items()}
