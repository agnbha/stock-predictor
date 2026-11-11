"""Saved training frames: one Parquet file per completed session.

Building one session's feature frame from stock-analyzer's views takes about
7 s, so every ``train`` spent ~13 minutes rebuilding 108 sessions that had not
changed. Reading one back takes a fraction of a second.

A saved frame holds features only. Labels are attached when it is loaded
(``models/training.py``), from the session's full minute candles, so training
for a different target or horizon reuses every saved frame.

A saved frame is used only if it was built from the same three things:

* **Variant**: the candidate inputs switched on as features, with their
  parameters. Each variant gets its own folder, so switching a feature on and
  off for a comparison keeps both sets.
* **Code**: a hash of the source files that build frames. Editing any of them
  rebuilds every frame, so nobody has to remember to bump a version.
* **Data**: a hash chained through every session up to and including this
  one, over its minute candles (count, first/last time, volume and price
  totals per symbol) and its daily candles. If stock-analyzer rewrites an old
  session, every later frame is rebuilt as well, because daily features and
  volume profiles look back.

Only completed sessions belong here. The session in progress changes every
minute; serving neither reads nor writes this store.

Layout::

    data/frames/<variant>/variant.json             what the variant is
    data/frames/<variant>/<session>.<key>.parquet  one session's frame
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

from stock_predictor.config import CONFIG
from stock_predictor.data.analyzer import AnalyzerViews

_PKG = Path(__file__).resolve().parents[1]

# Everything that turns stock-analyzer's views into a saved feature frame.
_FRAME_SOURCES = (
    "features",
    "data/analyzer.py",
    "data/framestore.py",
    "clock.py",
    "params.py",
)


def _sha(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode())
        h.update(b"\x00")
    return h.hexdigest()


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for rel in _FRAME_SOURCES:
        p = _PKG / rel
        files = sorted(p.rglob("*.py")) if p.is_dir() else [p]
        for f in files:
            h.update(str(f.relative_to(_PKG)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def data_fingerprints(views: AnalyzerViews) -> dict[str, str]:
    """Session -> hash of every session's data up to and including it."""
    by_session: dict[str, list] = {}
    for row in views.candle_digest():
        by_session.setdefault(row[0], []).append(tuple(row[1:]))
    daily: dict[str, list] = {}
    for c in views.daily_candles():
        daily.setdefault(c.session_date, []).append(
            (c.symbol, c.open, c.high, c.low, c.close, c.volume)
        )
    out: dict[str, str] = {}
    chain = ""
    for sd in sorted(set(by_session) | set(daily)):
        chain = _sha(chain, sd, by_session.get(sd, []), sorted(daily.get(sd, [])))
        out[sd] = chain
    return out


def variant_key(description: dict) -> str:
    return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()[:12]


class FrameStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or CONFIG.frames_dir)

    def _dir(self, variant: str, description: dict) -> Path:
        d = self.root / variant
        if not (d / "variant.json").exists():
            d.mkdir(parents=True, exist_ok=True)
            (d / "variant.json").write_text(json.dumps(description, indent=2, sort_keys=True))
        return d

    def path(self, variant: str, session_date: str, key: str) -> Path:
        return self.root / variant / f"{session_date}.{key[:16]}.parquet"

    def has(self, variant: str, session_date: str, key: str) -> bool:
        return self.path(variant, session_date, key).exists()

    def load(self, variant: str, session_date: str, key: str):
        p = self.path(variant, session_date, key)
        if not p.exists():
            return None
        import pandas as pd

        try:
            return pd.read_parquet(p)
        except Exception:  # a torn or foreign file is a miss, not an error
            return None

    def save(self, variant: str, description: dict, session_date: str, key: str, frame) -> Path:
        """Write atomically and drop this session's frames built from older
        code or data, so a variant holds one file per session."""
        d = self._dir(variant, description)
        p = self.path(variant, session_date, key)
        tmp = p.with_suffix(".tmp")
        frame.to_parquet(tmp, compression="zstd", index=False)
        os.replace(tmp, p)
        for old in d.glob(f"{session_date}.*.parquet"):
            if old != p:
                old.unlink(missing_ok=True)
        return p

    def usage(self) -> dict[str, dict[str, int]]:
        """Variant -> {"files": n, "bytes": total}."""
        out: dict[str, dict[str, int]] = {}
        if not self.root.exists():
            return out
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            files = list(d.glob("*.parquet"))
            out[d.name] = {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
        return out
