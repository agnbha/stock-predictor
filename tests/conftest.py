from __future__ import annotations

import os
from pathlib import Path

import pytest

from stock_predictor.config import CONFIG
from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.data.store import Store

_ANALYZER_DB = CONFIG.analyzer_db
_HAS_ANALYZER = _ANALYZER_DB.exists()

requires_analyzer = pytest.mark.skipif(
    not _HAS_ANALYZER, reason=f"stock-analyzer DB not present at {_ANALYZER_DB}"
)

# stock-analyzer checkout, for the feature-parity fixture
_ANALYZER_ROOT = _ANALYZER_DB.parent.parent
FIXTURE = _ANALYZER_ROOT / "src" / "test" / "resources" / "feature-parity.json"


@pytest.fixture(scope="session")
def views():
    if not _HAS_ANALYZER:
        pytest.skip("no analyzer DB")
    v = AnalyzerViews(_ANALYZER_DB)
    yield v
    v.close()


@pytest.fixture(scope="session")
def sessions(views):
    return views.completed_sessions()


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sp.db")
    yield s
    s.close()
