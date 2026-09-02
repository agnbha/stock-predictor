from __future__ import annotations

from stock_predictor.eval.splits import holdout_split, walk_forward_folds

SESSIONS = [f"2026-06-{d:02d}" for d in range(1, 29)]  # 28 fake sessions


def test_holdout_takes_the_final_sessions():
    dev, hold = holdout_split(SESSIONS, 5)
    assert hold == SESSIONS[-5:]
    assert dev == SESSIONS[:-5]
    assert set(dev).isdisjoint(hold)


def test_walk_forward_is_purged_and_ordered():
    folds = walk_forward_folds(SESSIONS, initial_train=10, test_size=5, embargo=1)
    assert folds
    for f in folds:
        assert max(f.train_sessions) < min(f.test_sessions)
        # embargo: at least one session gap between train end and test start
        gap_start = SESSIONS.index(min(f.test_sessions))
        train_end = SESSIONS.index(max(f.train_sessions))
        assert gap_start - train_end >= 2
        assert set(f.train_sessions).isdisjoint(f.test_sessions)


def test_expanding_train_grows():
    folds = walk_forward_folds(SESSIONS, initial_train=10, test_size=5, expanding=True)
    sizes = [len(f.train_sessions) for f in folds]
    assert sizes == sorted(sizes)
