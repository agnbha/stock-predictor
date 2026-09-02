"""The feature contract, checked against stock-analyzer's pinned fixture.

If this fails, this port and the Java ``FeatureExtractor`` disagree, and any
model trained on one and served through the other is scoring against the wrong
data.
"""

from __future__ import annotations

import json

import pytest

from stock_predictor.features.parity import FEATURE_NAMES, FeatureCandle, extract_features
from tests.conftest import FIXTURE

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason=f"no fixture at {FIXTURE}")

TOL = 1e-9


@pytest.fixture(scope="module")
def fixture():
    return json.loads(FIXTURE.read_text())


def test_feature_names_match_contract(fixture):
    assert list(FEATURE_NAMES) == fixture["feature_names"]


def test_contract_is_well_formed():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert all(n.replace("_", "").isalnum() and n.islower() for n in FEATURE_NAMES)


def test_vectors_reproduce_the_fixture(fixture):
    from datetime import date

    candles = [
        FeatureCandle(ts, o, h, l, c, v) for ts, o, h, l, c, v in fixture["candles"]
    ]
    prior = fixture["prior_session"]
    rows = extract_features(
        candles,
        date.fromisoformat(fixture["session_date"]),
        prior_close=prior["close"],
        prior_high=prior["high"],
    )
    assert len(rows) == len(fixture["expected"])
    for i, (got, want) in enumerate(zip(rows, fixture["expected"])):
        assert got["ts_epoch"] == want["ts_epoch"]
        for name, g, w in zip(FEATURE_NAMES, got["values"], want["values"]):
            assert abs(g - w) <= TOL, f"row {i}, feature {name}: {g} != {w}"
