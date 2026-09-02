"""Ties the pieces together for serving: data -> features -> model -> cache/store.

If a trained model is available it is used; otherwise the service falls back to
the ``highest_volatility`` baseline so ``/predictions/top`` still answers end to
end with nothing trained -- the same design as stock-analyzer's
``FallbackSignalModel`` (design section 7).
"""

from __future__ import annotations

import time
from datetime import date
from typing import Optional

from stock_predictor.config import CONFIG
from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.data.store import PredictionRow, Store
from stock_predictor.eval.baselines import run_baseline
from stock_predictor.labels.targets import build_targets
from stock_predictor.serving.cache import CachedRanking, PredictionCache, RankedPick

FALLBACK_VERSION = "fallback-highest-volatility/v1"


class PredictorService:
    def __init__(
        self,
        views: Optional[AnalyzerViews] = None,
        store: Optional[Store] = None,
        cache: Optional[PredictionCache] = None,
        model=None,
        model_version: Optional[str] = None,
    ):
        self.views = views or AnalyzerViews()
        self.store = store or Store()
        self.cache = cache or PredictionCache()
        self.model = model
        self.model_version = model_version or FALLBACK_VERSION

    # -- the schedule calls this -----------------------------------------

    def recompute(self, session_date: str, as_of: int) -> CachedRanking:
        picks = self._score(session_date, as_of)
        ranking = CachedRanking(
            as_of=as_of,
            computed_at=time.time(),
            model_version=self.model_version,
            session_date=session_date,
            picks=tuple(picks),
        )
        self.cache.put(ranking)
        self._persist(ranking)
        return ranking

    def _score(self, session_date: str, as_of: int) -> list[RankedPick]:
        if self.model is not None:
            try:
                return self._score_with_model(session_date, as_of)
            except Exception:
                pass  # fall through to baseline; never fail the request path
        baseline = run_baseline("highest_volatility", self.views, session_date, as_of, CONFIG.top_n)
        return [
            RankedPick(
                rank=i + 1, symbol=p.symbol,
                entry_window_start=None, entry_window_end=None, duration_bucket=None,
                gain_p25=0.0, gain_p50=float(p.gain_p50), gain_p75=0.0, score=float(p.gain_p50),
            )
            for i, p in enumerate(baseline)
        ]

    def _score_with_model(self, session_date: str, as_of: int) -> list[RankedPick]:
        from stock_predictor.features.pipeline import build_feature_frame, latest_row_per_symbol
        from stock_predictor.models.aggregate import PerMinutePrediction, aggregate_to_topn

        frame = build_feature_frame(self.views, session_date, as_of)
        if frame.empty:
            raise RuntimeError("no features")
        pred = self.model.predict(frame)
        per_minute = [
            PerMinutePrediction(
                symbol=frame.iloc[i]["symbol"], ts_epoch=int(frame.iloc[i]["ts_epoch"]),
                p_up=float(pred.p_up[i]), gain_p25=float(pred.gain_p25[i]),
                gain_p50=float(pred.gain_p50[i]), gain_p75=float(pred.gain_p75[i]),
            )
            for i in range(len(frame))
        ]
        top = aggregate_to_topn(per_minute, CONFIG.top_n)
        return [
            RankedPick(
                rank=i + 1, symbol=s.symbol,
                entry_window_start=s.entry_window_start, entry_window_end=s.entry_window_end,
                duration_bucket=s.duration_bucket,
                gain_p25=s.gain_p25, gain_p50=s.gain_p50, gain_p75=s.gain_p75, score=s.score,
            )
            for i, s in enumerate(top)
        ]

    def _persist(self, ranking: CachedRanking) -> None:
        rows = [
            PredictionRow(
                model_version=ranking.model_version, session_date=ranking.session_date,
                as_of=ranking.as_of, computed_at=int(ranking.computed_at), rank=p.rank,
                symbol=p.symbol, entry_window_start=p.entry_window_start,
                entry_window_end=p.entry_window_end, duration_bucket=p.duration_bucket,
                gain_p25=p.gain_p25, gain_p50=p.gain_p50, gain_p75=p.gain_p75, score=p.score,
            )
            for p in ranking.picks
        ]
        self.store.save_predictions(rows)

    # -- next-day scoring ------------------------------------------------

    def score_yesterday(self, today: str) -> int:
        """Fill realised gains on predictions whose session has since completed."""
        pending = self.store.unscored_predictions(before_session=today)
        n = 0
        for row in pending:
            targets = build_targets(self.views, row["session_date"], row["as_of"], top_n=99)
            by_sym = {t.symbol: t for t in targets}
            t = by_sym.get(row["symbol"])
            realised = t.gain_pct if t else 0.0
            dur = t.duration_minutes if t else None
            self.store.score_prediction(row["id"], realised, dur, int(time.time()))
            n += 1
        return n

    # -- stock-analyzer's per-candle contract --------------------------

    def predict_intraday(self, symbol: str, session_date: str, horizon_minutes: int,
                         feature_names: list[str], rows: list[dict]) -> dict:
        preds = []
        can_score = self.model is not None and set(feature_names).issuperset(
            set(getattr(self.model, "feature_names", []))
        )
        for r in rows:
            if not can_score:
                preds.append({"ts_epoch": r["ts_epoch"], "signal": "NEUTRAL", "probability": 0.0,
                              "reason": "model not loaded"})
                continue
            idx = {n: i for i, n in enumerate(feature_names)}
            vec = [[r["values"][idx[n]] for n in self.model.feature_names]]
            p_up = float(self.model._clf.predict_proba(vec)[0])
            signal = "ENTRY" if p_up >= 0.5 else "NEUTRAL"
            preds.append({"ts_epoch": r["ts_epoch"], "signal": signal, "probability": p_up,
                          "reason": None})
        self.store.save_intraday_signals(
            [
                {"model_version": self.model_version, "symbol": symbol, "session_date": session_date,
                 "ts_epoch": p["ts_epoch"], "horizon_minutes": horizon_minutes,
                 "signal": p["signal"], "probability": p["probability"], "reason": p["reason"]}
                for p in preds
            ]
        )
        return {"model_version": self.model_version, "predictions": preds}
