from __future__ import annotations

from datetime import date

import pytest

from stock_predictor.clock import NSE
from stock_predictor.config import OPENING_RANGE_T1
from tests.conftest import requires_analyzer

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(views, store, tmp_path):
    from stock_predictor.serving.app import create_app
    from stock_predictor.serving.cache import PredictionCache
    from stock_predictor.serving.service import PredictorService

    svc = PredictorService(views=views, store=store, cache=PredictionCache())
    return svc, TestClient(create_app(svc))


@requires_analyzer
def test_health_before_and_after_recompute(client, sessions):
    svc, c = client
    assert c.get("/health").json()["as_of"] is None
    assert c.get("/predictions/top").status_code == 503

    sd = sessions[45]
    svc.recompute(sd, NSE.epoch_of(date.fromisoformat(sd), OPENING_RANGE_T1))

    h = c.get("/health").json()
    assert h["status"] == "ok" and h["as_of"] is not None

    top = c.get("/predictions/top?n=5").json()
    assert top["session_date"] == sd
    assert 1 <= len(top["picks"]) <= 5
    assert top["picks"][0]["rank"] == 1
    assert "as_of" in top

    sym = top["picks"][0]["symbol"]
    one = c.get(f"/predictions/{sym}").json()
    assert one["symbol"] == sym
    assert c.get("/predictions/NOTASYMBOL").status_code == 404


@requires_analyzer
def test_model_endpoint_reports_fallback(client):
    _svc, c = client
    m = c.get("/model").json()
    assert m["model_version"]
    assert m.get("trained") is False


@requires_analyzer
def test_intraday_contract_endpoint_shape(client):
    _svc, c = client
    from stock_predictor.features.parity import FEATURE_NAMES

    body = {
        "symbol": "RELIANCE",
        "session_date": "2026-08-27",
        "horizon_minutes": 30,
        "feature_names": list(FEATURE_NAMES),
        "rows": [{"ts_epoch": 1787802300, "values": [0.0] * len(FEATURE_NAMES)}],
    }
    r = c.post("/predict/intraday", json=body).json()
    assert "model_version" in r
    assert r["predictions"][0]["signal"] in {"ENTRY", "EXIT", "NEUTRAL"}
