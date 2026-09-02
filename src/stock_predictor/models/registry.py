"""Model artefacts and versioning.

A model version is ``<backend>-triple-barrier/<trained-through-date>`` so a
served answer always names the exact model behind it (design section 7, the
``/model`` endpoint and the ``model_version`` on every stored prediction).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from stock_predictor.config import CONFIG


def model_version_string(backend: str, trained_through: str) -> str:
    return f"{backend}-triple-barrier/{trained_through}"


@dataclass
class ModelCard:
    model_version: str
    backend: str
    trained_through: str
    train_sessions: int
    feature_names: list[str]
    holdout_metrics: dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ModelRegistry:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or CONFIG.artifacts_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, version: str) -> Path:
        return self.root / version.replace("/", "__")

    def save(self, version: str, model: Any, card: ModelCard) -> Path:
        d = self._dir(version)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "model.pkl").open("wb") as fh:
            pickle.dump(model, fh)
        with (d / "card.json").open("w") as fh:
            json.dump(asdict(card), fh, indent=2)
        return d

    def load(self, version: str) -> tuple[Any, ModelCard]:
        d = self._dir(version)
        with (d / "model.pkl").open("rb") as fh:
            model = pickle.load(fh)
        with (d / "card.json").open() as fh:
            card = ModelCard(**json.load(fh))
        return model, card

    def latest(self) -> Optional[str]:
        cards = sorted(self.root.glob("*/card.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cards:
            return None
        with cards[0].open() as fh:
            return json.load(fh)["model_version"]

    def card(self, version: str) -> Optional[ModelCard]:
        p = self._dir(version) / "card.json"
        if not p.exists():
            return None
        with p.open() as fh:
            return ModelCard(**json.load(fh))
