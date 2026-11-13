"""Model artefacts and versioning.

A model version is ``<backend>-triple-barrier/<trained-through-date>`` (or
``<backend>-target-1pct-40m/<date>`` for a fixed target) so a served answer
always names the exact model behind it (design section 7, the ``/model``
endpoint and the ``model_version`` on every stored prediction).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from stock_predictor.config import CONFIG


def model_version_string(backend: str, trained_through: str, target=None, minute_step: int = 1,
                         recent_days: Optional[int] = None, next_session: bool = False) -> str:
    """``target`` is the TripleBarrierConfig trained on; a fixed target names
    itself so models for different targets do not overwrite each other. A
    training window of only the last ``recent_days`` and a minute step other
    than 1 are named too, so they sit beside the full-history model."""
    if next_session and target is not None and target.fixed:
        label = f"nextday-{target.target_pct:g}pct"
    elif target is not None and target.fixed:
        stop = f"-stop{target.stop_pct:g}pct" if target.stop_pct is not None else ""
        window = "close" if target.to_close else f"{target.horizon_minutes}m"
        label = f"target-{target.target_pct:g}pct{stop}-{window}"
    elif target is not None and target.horizon_minutes != 30:
        label = f"triple-barrier-{target.horizon_minutes}m"
    else:
        label = "triple-barrier"
    recent = f"-recent{recent_days}d" if recent_days else ""
    step = f"-m{minute_step}" if minute_step != 1 else ""
    return f"{backend}-{label}{recent}{step}/{trained_through}"


@dataclass
class ModelCard:
    model_version: str
    backend: str
    trained_through: str
    train_sessions: int
    feature_names: list[str]
    holdout_metrics: dict[str, float] = field(default_factory=dict)
    # The params the model was trained with -- candidate `feature` switches
    # decide its columns, so serving needs the same ones (design 4.1).
    params: dict[str, Any] = field(default_factory=dict)
    # Training kept every k-th minute (training_set); 1 = every minute.
    minute_step: int = 1
    # The label trained on (TripleBarrierConfig as a dict); empty = the default.
    target: dict[str, Any] = field(default_factory=dict)
    # For a fixed target: P(target) as predicted vs as seen, on sessions the
    # calibration model had not trained on (rows: band, predicted, seen, rows).
    calibration: list[dict[str, Any]] = field(default_factory=list)
    # What the model's p25/p50/p75 band describes (ForwardModel.band).
    band: str = "forward_return_pct"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # The training window: first session, and "only the last N days" when set.
    trained_from: str = ""
    recent_days: Optional[int] = None
    train_rows: int = 0
    # Where a prediction's gain is measured from: each minute's close
    # ("minute"), or the next session's open ("next_open", models/nextday.py).
    entry: str = "minute"

    def recipe(self) -> dict[str, Any]:
        """What makes two models the same kind of model: a result kept beside
        one (a held-out evaluation) describes the others."""
        return {"target": self.target, "band": self.band, "recent_days": self.recent_days,
                "minute_step": self.minute_step, "entry": self.entry}


def card_target(card: ModelCard):
    from stock_predictor.models.triple_barrier import TripleBarrierConfig

    return TripleBarrierConfig(**card.target) if card.target else TripleBarrierConfig()


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

    def save_extra(self, version: str, name: str, data: Any) -> Path:
        """A JSON result kept beside a model: ``evaluation``, ``screener_evidence``."""
        p = self._dir(version) / f"{name}.json"
        with p.open("w") as fh:
            json.dump(data, fh, indent=1)
        return p

    def load_extra(self, version: str, name: str) -> Optional[Any]:
        p = self._dir(version) / f"{name}.json"
        if not p.exists():
            return None
        with p.open() as fh:
            return json.load(fh)

    def cards(self) -> list[ModelCard]:
        """Every saved model's card, newest first."""
        out = []
        for p in sorted(self.root.glob("*/card.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            with p.open() as fh:
                out.append(ModelCard(**json.load(fh)))
        return out

    def latest_with_target(self, fixed: bool = True) -> Optional[str]:
        """The full-history fixed-target model (train --target) trained through
        the latest session; of those, the one saved last."""
        found = [c for c in self.cards()
                 if bool(c.target and c.target.get("target_pct") is not None) == fixed
                 and not c.recent_days and c.entry == "minute"]
        if not found:
            return None
        return max(found, key=lambda c: c.trained_through).model_version  # cards() is newest first

    def extra_for_target(self, name: str, target: dict) -> Optional[tuple[str, Any]]:
        """The newest ``name`` result kept beside any model with this target:
        the held-out evaluation of a model trained without the held-out month
        describes the live model trained on the same target with it."""
        for c in self.cards():
            if c.target == target:
                data = self.load_extra(c.model_version, name)
                if data is not None:
                    return c.model_version, data
        return None

    def extra_like(self, card: ModelCard, name: str) -> Optional[tuple[str, Any]]:
        """The newest ``name`` result kept beside any model with ``card``'s
        recipe (target, band, window, minute step), the model itself included."""
        for c in self.cards():
            if c.recipe() == card.recipe():
                data = self.load_extra(c.model_version, name)
                if data is not None:
                    return c.model_version, data
        return None

    def prune(self, keep: int = 3, dry_run: bool = False) -> list[str]:
        """Delete superseded models: of each recipe, keep the ``keep`` trained
        through the latest sessions, and the newest one holding each kind of
        kept result (a held-out check, screener evidence), since live models
        find their evidence through it. Returns the versions removed."""
        import shutil

        groups: dict[str, list[ModelCard]] = {}
        for c in self.cards():
            groups.setdefault(json.dumps(c.recipe(), sort_keys=True), []).append(c)
        gone = []
        for cards in groups.values():
            cards.sort(key=lambda c: (c.trained_through, c.created_at), reverse=True)
            protected = {c.model_version for c in cards[:keep]}
            for name in ("evaluation", "screener_evidence"):
                holder = next((c for c in cards if (self._dir(c.model_version) / f"{name}.json").exists()), None)
                if holder is not None:
                    protected.add(holder.model_version)
            for c in cards:
                if c.model_version in protected:
                    continue
                gone.append(c.model_version)
                if not dry_run:
                    shutil.rmtree(self._dir(c.model_version))
        return gone

    def card(self, version: str) -> Optional[ModelCard]:
        p = self._dir(version) / "card.json"
        if not p.exists():
            return None
        with p.open() as fh:
            return ModelCard(**json.load(fh))
