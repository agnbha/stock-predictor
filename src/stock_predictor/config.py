"""Runtime configuration: where the two databases live, and a few constants.

Every path can be overridden by an environment variable so the same code runs
in a check-out, in CI, and in a container without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


@dataclass(frozen=True)
class Config:
    # Read-only. stock-analyzer's SQLite database (its published views).
    analyzer_db: Path = field(
        default_factory=lambda: _env_path(
            "STOCK_ANALYZER_DB", _REPO_ROOT.parent / "stock-analyzer" / "data" / "stock-analyzer.db"
        )
    )
    # stock-predictor's own store: predictions, feature snapshots, metrics.
    store_db: Path = field(
        default_factory=lambda: _env_path(
            "STOCK_PREDICTOR_DB", _REPO_ROOT / "data" / "stock-predictor.db"
        )
    )
    # Trained model artefacts.
    artifacts_dir: Path = field(
        default_factory=lambda: _env_path("STOCK_PREDICTOR_ARTIFACTS", _REPO_ROOT / "artifacts")
    )

    # The opportunity detector version to learn from. stock-analyzer versions its
    # ground truth so a rules change produces a parallel series (design section 1).
    detector_version: str = os.environ.get("STOCK_PREDICTOR_DETECTOR", "topk-nonoverlap/highlow/v1")

    # How many windows a prediction contains.
    top_n: int = int(os.environ.get("STOCK_PREDICTOR_TOP_N", "5"))

    # Held-out final stretch, never touched during development (design section 6).
    holdout_sessions: int = int(os.environ.get("STOCK_PREDICTOR_HOLDOUT", "21"))

    def ensure_dirs(self) -> None:
        self.store_db.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


CONFIG = Config()

# Prediction points (design section 2). T0 pre-open, T1 after the opening range,
# then every 15 minutes until 15:00.
PREOPEN_T0 = time(9, 0)
OPENING_RANGE_T1 = time(9, 30)
INTRADAY_STEP_MINUTES = 15
LAST_INTRADAY_POINT = time(15, 0)

# Duration buckets for the classifier (design section 1).
DURATION_BUCKETS = ("<15m", "15-60m", "60-180m", ">180m")


def duration_bucket(minutes: float) -> str:
    if minutes < 15:
        return "<15m"
    if minutes < 60:
        return "15-60m"
    if minutes < 180:
        return "60-180m"
    return ">180m"
