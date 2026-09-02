"""FastAPI surface (design section 7).

    GET  /health                 liveness, model version, cache age
    GET  /predictions/top?n=5    the ranking, with bands and as_of
    GET  /predictions/{symbol}   one symbol's expectation
    GET  /model                  version, trained window, held-out metrics
    POST /predict/intraday       stock-analyzer's existing per-candle contract

Every response that carries a ranking also carries ``as_of`` -- a consumer must
be able to tell a fresh answer from a stale one.
"""

# NB: no `from __future__ import annotations` here -- FastAPI/pydantic need the
# request-model annotations as real objects, not strings.

from typing import List, Optional

try:
    from pydantic import BaseModel

    _HAVE_PYDANTIC = True
except ModuleNotFoundError:  # pragma: no cover
    _HAVE_PYDANTIC = False
    BaseModel = object  # type: ignore

from stock_predictor.serving.service import PredictorService


class IntradayRow(BaseModel):
    ts_epoch: int
    values: List[float]


class IntradayRequest(BaseModel):
    symbol: str
    session_date: str
    horizon_minutes: int = 30
    feature_names: List[str]
    rows: List[IntradayRow]


def create_app(service: Optional[PredictorService] = None):
    try:
        from fastapi import FastAPI, HTTPException
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("serving needs the extra: pip install 'stock-predictor[serving]'") from exc

    svc = service or PredictorService()
    app = FastAPI(title="stock-predictor", version="0.1.0")

    def _ranking_or_503():
        r = svc.cache.get()
        if r is None:
            raise HTTPException(status_code=503, detail="no prediction computed yet")
        return r

    @app.get("/health")
    def health():
        r = svc.cache.get()
        return {
            "status": "ok",
            "model_version": svc.model_version,
            "cache_age_seconds": r.age_seconds() if r else None,
            "as_of": r.as_of if r else None,
        }

    @app.get("/predictions/top")
    def predictions_top(n: int = 5):
        r = _ranking_or_503()
        return {
            "as_of": r.as_of,
            "session_date": r.session_date,
            "model_version": r.model_version,
            "picks": [
                {
                    "rank": p.rank,
                    "symbol": p.symbol,
                    "entry_window": {"start": p.entry_window_start, "end": p.entry_window_end},
                    "duration_bucket": p.duration_bucket,
                    "gain_pct": {"p25": p.gain_p25, "p50": p.gain_p50, "p75": p.gain_p75},
                    "score": p.score,
                }
                for p in r.picks[:n]
            ],
        }

    @app.get("/predictions/{symbol}")
    def prediction_for(symbol: str):
        r = _ranking_or_503()
        p = svc.cache.for_symbol(symbol.upper())
        if p is None:
            raise HTTPException(status_code=404, detail=f"{symbol} not in the current ranking")
        return {
            "as_of": r.as_of,
            "session_date": r.session_date,
            "model_version": r.model_version,
            "symbol": p.symbol,
            "rank": p.rank,
            "entry_window": {"start": p.entry_window_start, "end": p.entry_window_end},
            "duration_bucket": p.duration_bucket,
            "gain_pct": {"p25": p.gain_p25, "p50": p.gain_p50, "p75": p.gain_p75},
        }

    @app.get("/model")
    def model_info():
        from stock_predictor.models.registry import ModelRegistry

        card = ModelRegistry().card(svc.model_version)
        if card is None:
            return {
                "model_version": svc.model_version,
                "trained": False,
                "note": "fallback baseline; nothing trained",
            }
        return {
            "model_version": card.model_version,
            "backend": card.backend,
            "trained_through": card.trained_through,
            "train_sessions": card.train_sessions,
            "holdout_metrics": card.holdout_metrics,
        }

    @app.post("/predict/intraday")
    def predict_intraday(req: IntradayRequest):
        return svc.predict_intraday(
            req.symbol,
            req.session_date,
            req.horizon_minutes,
            req.feature_names,
            [row.model_dump() for row in req.rows],
        )

    return app
