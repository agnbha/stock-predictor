"""Fundamentals score, 0-100 per quarter (design section 3.3).

**No data source exists yet.** stock-analyzer has no filings or revenue data,
and the Groww API does not serve financial statements. So this is an interface
with a null implementation: everything else is built and evaluated without it,
and a real provider slots in behind :class:`FundamentalScore` later.

A caution from the design: quarterly fundamentals move four times a year and
intraday windows resolve in hours. This feature is expected to shift the
*ranking between symbols* slightly and to say almost nothing about *today*.
Build it for that modest role.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class FundamentalScore(Protocol):
    def score(self, symbol: str, as_of: date) -> Optional[float]:
        """A 0-100 quality score effective at ``as_of``, or ``None`` if unknown."""
        ...


class NullFundamentalScore:
    """The default. Always ``None`` -- the model simply omits the feature."""

    def score(self, symbol: str, as_of: date) -> Optional[float]:  # noqa: ARG002
        return None


class CsvFundamentalScore:
    """Bring-your-own: a CSV of ``symbol,effective_date,score`` rows from a
    screener-style provider (the fastest of the design's candidate sources).

    ``score(symbol, as_of)`` returns the most recent row with
    ``effective_date <= as_of`` -- point-in-time, no look-ahead.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._by_symbol: dict[str, list[tuple[date, float]]] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    d = date.fromisoformat(row["effective_date"])
                    s = float(row["score"])
                except (KeyError, ValueError):
                    continue
                self._by_symbol.setdefault(row["symbol"].strip(), []).append((d, s))
        for rows in self._by_symbol.values():
            rows.sort()

    def score(self, symbol: str, as_of: date) -> Optional[float]:
        rows = self._by_symbol.get(symbol)
        if not rows:
            return None
        eligible = [s for d, s in rows if d <= as_of]
        return eligible[-1] if eligible else None


def get_default(csv_path: Optional[Path] = None) -> FundamentalScore:
    if csv_path and Path(csv_path).exists():
        return CsvFundamentalScore(csv_path)
    return NullFundamentalScore()
