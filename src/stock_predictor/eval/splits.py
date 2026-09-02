"""Purged walk-forward splitting.

Design section 6: split by day, never within one -- all symbols of a session go
to the same fold. Labels look forward within a day, so an **embargo** of one
session either side of each boundary is purged from training. There are 66
folds' worth of *boundaries* and over a million *examples*; the day-level split
is about leakage, not sample size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Fold({self.fold_id}: train={len(self.train_sessions)}d "
            f"[{self.train_sessions[0]}..{self.train_sessions[-1]}], "
            f"test={len(self.test_sessions)}d [{self.test_sessions[0]}..{self.test_sessions[-1]}])"
        )


def holdout_split(sessions: Sequence[str], holdout: int) -> tuple[list[str], list[str]]:
    """The final ``holdout`` sessions are set aside and never touched during
    development (design section 6). Returns ``(development, holdout)``."""
    s = sorted(sessions)
    if holdout <= 0:
        return list(s), []
    if holdout >= len(s):
        raise ValueError(f"holdout={holdout} >= {len(s)} sessions")
    return s[:-holdout], s[-holdout:]


def walk_forward_folds(
    sessions: Sequence[str],
    *,
    initial_train: int = 20,
    test_size: int = 5,
    step: int | None = None,
    embargo: int = 1,
    expanding: bool = True,
) -> list[Fold]:
    """Expanding (or rolling) walk-forward folds over an ordered session list.

    ``embargo`` sessions immediately before each test block are purged from
    training so a forward-looking label cannot straddle the boundary.
    """
    s = sorted(sessions)
    if step is None:
        step = test_size
    folds: list[Fold] = []
    start_test = initial_train + embargo
    k = 1
    while start_test + test_size <= len(s):
        test = s[start_test : start_test + test_size]
        train_end = start_test - embargo
        train_start = 0 if expanding else max(0, train_end - initial_train)
        train = s[train_start:train_end]
        if train:
            folds.append(
                Fold(fold_id=f"wf{k:02d}", train_sessions=tuple(train), test_sessions=tuple(test))
            )
            k += 1
        start_test += step
    return folds
