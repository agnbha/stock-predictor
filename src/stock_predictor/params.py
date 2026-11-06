"""Tunable parameters for the candidate screening inputs (design section 4.1).

Seven inputs were proposed for picking stocks -- an opening check at 09:45,
run-up caps, RSI limits, volume, where the stock closed in its range, score
order, and nearness to the 120-day high. Tested elsewhere on daily and
next-day horizons, most showed no reliable effect. They are kept here as named,
configurable parameters so they can be measured on *this* project's target (the
best window still open after T) rather than accepted or dropped on faith.

Each input has two independent switches:

* ``feature`` -- add its value as a stage-1 model column (soft: the trees may
  ignore it);
* ``filter``  -- drop symbols that fail its rule before ranking (hard: it
  changes which symbols can be picked at all).

Both default to **off**. ``stock-predictor ablate`` measures every input with
the switches as they stand, and turning one on is a one-line change in a
params file once it has earned its place (design section 10: prefer the simpler
model whenever the difference is within noise).

Defaults live in this module. Override them with a TOML or JSON file, via
``--params FILE`` or the ``STOCK_PREDICTOR_PARAMS`` environment variable, and
with single values via ``--set section.key=value``. See ``params.example.toml``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import time
from pathlib import Path
from typing import Any, Mapping, Optional

RANGE_STATES = ("below", "inside", "above")
RANK_BY = ("score", "risk_adjusted")


@dataclass(frozen=True)
class OpeningCheck:
    """Opening-range and VWAP state at ``check_time``.

    The opening range is the high/low of the first ``opening_range_minutes``
    after 09:15. At ``check_time`` the last price is ``below`` that range,
    ``inside`` it, or ``above`` it, and above or below the session VWAP. As with
    every as-of in this project, the candle stamped ``check_time`` is included.

    Only usable when the prediction point is at or after ``check_time``; before
    that the state is unknown and the rule does not apply.
    """

    feature: bool = False
    filter: bool = False
    check_time: str = "09:45"
    opening_range_minutes: int = 15
    allowed_range_states: tuple = ("above",)
    require_above_vwap: bool = True

    @property
    def check_clock(self) -> time:
        hh, mm = self.check_time.split(":")
        return time(int(hh), int(mm))


@dataclass(frozen=True)
class RunUpCap:
    """Close-to-close gain over the last ``lookback_days`` completed sessions.
    The filter drops symbols that have already run up more than ``max_pct``."""

    feature: bool = False
    filter: bool = False
    lookback_days: int = 5
    max_pct: float = 10.0


@dataclass(frozen=True)
class RsiLimits:
    """RSI over ``period`` completed sessions (same simple-average definition as
    stock-analyzer's ``v_symbol_trend``). The filter keeps ``min <= RSI <= max``."""

    feature: bool = False
    filter: bool = False
    period: int = 14
    min: float = 30.0
    max: float = 70.0


@dataclass(frozen=True)
class VolumeCheck:
    """Prior session's volume relative to the mean of the ``lookback_days``
    sessions before it. The filter keeps ``relative volume >= min_relative_volume``."""

    feature: bool = False
    filter: bool = False
    lookback_days: int = 20
    min_relative_volume: float = 1.0


@dataclass(frozen=True)
class CloseLocation:
    """Where the prior session closed in its own range:
    ``(close - low) / (high - low)``, 0 = at the low, 1 = at the high.
    The filter keeps ``min <= location <= max``."""

    feature: bool = False
    filter: bool = False
    min: float = 0.5
    max: float = 1.0


@dataclass(frozen=True)
class ScoreOrder:
    """How the final ranking is ordered, and how its risk is measured.

    ``rank_by="score"`` ranks on stage 2's discounted expected move.
    ``"risk_adjusted"`` divides that by the symbol's ATR % first, so a pick is
    not ranked high merely for being volatile. ``adverse_move_pct`` sets the
    ``adverse_rate`` metric: the share of picks whose price falls at least this
    far below its as-of price later in the session.
    """

    rank_by: str = "score"
    adverse_move_pct: float = 2.0


@dataclass(frozen=True)
class MediumTermMomentum:
    """Distance of the prior close below its ``lookback_days`` high, in % (0 = at
    the high). Unavailable until that many completed sessions of daily history
    exist. The filter keeps symbols within ``max_dist_from_high_pct`` of the high."""

    feature: bool = False
    filter: bool = False
    lookback_days: int = 120
    max_dist_from_high_pct: float = 10.0


@dataclass(frozen=True)
class Params:
    opening_check: OpeningCheck = field(default_factory=OpeningCheck)
    run_up: RunUpCap = field(default_factory=RunUpCap)
    rsi: RsiLimits = field(default_factory=RsiLimits)
    volume: VolumeCheck = field(default_factory=VolumeCheck)
    close_location: CloseLocation = field(default_factory=CloseLocation)
    score_order: ScoreOrder = field(default_factory=ScoreOrder)
    momentum: MediumTermMomentum = field(default_factory=MediumTermMomentum)

    def __post_init__(self) -> None:
        bad = set(self.opening_check.allowed_range_states) - set(RANGE_STATES)
        if bad:
            raise ValueError(f"opening_check.allowed_range_states: unknown {sorted(bad)}; "
                             f"use {RANGE_STATES}")
        self.opening_check.check_clock  # noqa: B018 -- validates HH:MM
        if self.score_order.rank_by not in RANK_BY:
            raise ValueError(f"score_order.rank_by must be one of {RANK_BY}")
        for name in ("run_up", "rsi", "volume", "momentum"):
            section = getattr(self, name)
            for f in fields(section):
                if f.name.endswith(("_days", "period")) and getattr(section, f.name) < 1:
                    raise ValueError(f"{name}.{f.name} must be >= 1")

    # -- views -------------------------------------------------------------

    def candidates(self) -> dict[str, Any]:
        """The six inputs that have feature/filter switches (score order has neither)."""
        return {f.name: getattr(self, f.name) for f in fields(self)
                if hasattr(getattr(self, f.name), "filter")}

    def active_filters(self) -> list[str]:
        return [name for name, sec in self.candidates().items() if sec.filter]

    def active_features(self) -> list[str]:
        return [name for name, sec in self.candidates().items() if sec.feature]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["opening_check"]["allowed_range_states"] = list(self.opening_check.allowed_range_states)
        return d

    # -- construction --------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Params":
        base = cls()
        sections = {f.name: f for f in fields(cls)}
        unknown = set(data) - set(sections)
        if unknown:
            raise ValueError(f"unknown params section(s): {sorted(unknown)}; known: {sorted(sections)}")
        updates = {}
        for name, values in data.items():
            current = getattr(base, name)
            updates[name] = _update_section(name, current, values)
        return replace(base, **updates)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Params":
        """Apply ``{"section.key": value}`` overrides, e.g. from ``--set``."""
        nested: dict[str, dict] = {}
        for dotted, value in overrides.items():
            if "." not in dotted:
                raise ValueError(f"override {dotted!r} must look like section.key")
            section, key = dotted.split(".", 1)
            nested.setdefault(section, {})[key] = value
        merged = self.to_dict()
        for section, values in nested.items():
            if section not in merged:
                raise ValueError(f"unknown params section {section!r}")
            merged[section].update(values)
        return Params.from_dict(merged)


def _update_section(name: str, current: Any, values: Mapping[str, Any]) -> Any:
    if not is_dataclass(current):
        raise ValueError(f"{name} is not a params section")
    known = {f.name: f for f in fields(current)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"unknown key(s) in [{name}]: {sorted(unknown)}; known: {sorted(known)}")
    coerced = {k: _coerce(f"{name}.{k}", getattr(current, k), v) for k, v in values.items()}
    return replace(current, **coerced)


def _coerce(path: str, default: Any, value: Any) -> Any:
    """Coerce a loaded or ``--set`` value to the type of its default."""
    if isinstance(default, bool):
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes", "on"):
                return True
            if low in ("false", "0", "no", "off"):
                return False
            raise ValueError(f"{path}: expected a boolean, got {value!r}")
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, tuple):
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        return tuple(value)
    return str(value)


def parse_set(items: Optional[list[str]]) -> dict[str, str]:
    """``["rsi.filter=true", "rsi.max=65"]`` -> ``{"rsi.filter": "true", ...}``."""
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set {item!r}: expected section.key=value")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_params(path: Optional[os.PathLike] = None) -> Params:
    """Defaults, overlaid with ``path`` (or ``$STOCK_PREDICTOR_PARAMS``) if given."""
    raw = path or os.environ.get("STOCK_PREDICTOR_PARAMS")
    if not raw:
        return Params()
    p = Path(raw).expanduser()
    text = p.read_text()
    if p.suffix == ".json":
        data = json.loads(text)
    elif p.suffix == ".toml":
        try:
            import tomllib  # type: ignore[import-not-found]  # Python 3.11+
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]
        data = tomllib.loads(text)
    else:
        raise ValueError(f"params file must be .toml or .json, got {p.name}")
    return Params.from_dict(data)


# Process-wide current params. The CLI sets it once at start-up; library code
# takes an explicit ``params=`` argument and falls back to this.
_CURRENT: Optional[Params] = None


def get_params() -> Params:
    global _CURRENT
    if _CURRENT is None:
        _CURRENT = load_params()
    return _CURRENT


def set_params(params: Params) -> None:
    global _CURRENT
    _CURRENT = params
