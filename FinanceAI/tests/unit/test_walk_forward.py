"""
§31 — proves the walk-forward split is a genuine expanding window: every
fold's validation window is strictly AFTER its training window (no overlap,
no gap that would let a fold peek), and later folds only ever grow the
training window forward in time, never shuffle.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipelines.train_three_models import _make_folds


class TestWalkForwardFolds:
    def test_folds_are_chronologically_non_overlapping(self):
        folds = _make_folds(pd.Timestamp("2021-01-01"), pd.Timestamp("2026-01-01"))
        assert len(folds) >= 2, "expected multiple folds over a 5-year synthetic range"
        for train_end, val_start, val_end in folds:
            assert train_end == val_start, "no gap and no overlap: training must end exactly where validation starts"
            assert val_start < val_end, "validation window must have positive length"

    def test_training_window_only_grows_forward(self):
        """Each fold's train_end must be strictly after the previous fold's — an expanding window never shrinks or resets."""
        folds = _make_folds(pd.Timestamp("2021-01-01"), pd.Timestamp("2026-01-01"))
        train_ends = [f[0] for f in folds]
        assert train_ends == sorted(train_ends), "train_end must be monotonically increasing across folds"
        assert len(set(train_ends)) == len(train_ends), "each fold must have a distinct, later training cutoff"

    def test_validation_windows_do_not_overlap_each_other(self):
        folds = _make_folds(pd.Timestamp("2021-01-01"), pd.Timestamp("2026-01-01"))
        for i in range(len(folds) - 1):
            _, _, val_end_i = folds[i]
            _, val_start_next, _ = folds[i + 1]
            assert val_end_i <= val_start_next, "consecutive validation windows must not overlap"

    def test_insufficient_range_produces_no_folds(self):
        """A range too short for even one fold must return an empty list, not a degenerate/invalid fold."""
        folds = _make_folds(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01"))
        assert folds == []
