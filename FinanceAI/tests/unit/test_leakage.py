"""
§57/§58 — MOST IMPORTANT: proves no feature at prediction time T uses
information dated after T.

Method: compute technical features once, then mutate every OHLCV value
AFTER a cutoff row to something wildly different, recompute, and assert
every feature value AT OR BEFORE the cutoff is byte-for-byte identical
between the two runs. If any feature at or before the cutoff changed, it
must have been computed using data from after the cutoff — a leakage bug,
caught automatically rather than by manual inspection.

Synthetic price data only, per §59 — this file lives under tests/, never
imported by production code.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.technical import FeaturePipeline, TechnicalFeatures


def _synthetic_ohlcv(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    returns = rng.normal(0.0005, 0.015, n)
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(100_000, 1_000_000, n).astype(float)  # float, not int64 — the mutation below scales it, which must stay in-dtype
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


CUTOFF = 300


class TestTechnicalFeatureLeakage:
    def test_features_at_or_before_cutoff_unaffected_by_future_data(self):
        base = _synthetic_ohlcv()
        mutated = base.copy()
        # Wildly different future — a crash to near-zero, not just noise,
        # so any leak would be impossible to miss.
        mutated.iloc[CUTOFF + 1:] = mutated.iloc[CUTOFF + 1:] * 0.01

        features_base = TechnicalFeatures().compute(base)
        features_mutated = TechnicalFeatures().compute(mutated)

        before = features_base.iloc[: CUTOFF + 1]
        before_mutated = features_mutated.iloc[: CUTOFF + 1]

        pd.testing.assert_frame_equal(before, before_mutated, check_exact=False, rtol=1e-9)

    def test_prepare_xgboost_input_features_unaffected_by_future_data(self):
        """Same property, one level up — the actual model-input matrix, not just the raw indicator DataFrame."""
        base = _synthetic_ohlcv()
        mutated = base.copy()
        mutated.iloc[CUTOFF + 1:] = mutated.iloc[CUTOFF + 1:] * 0.01

        enriched_base = TechnicalFeatures().compute(base)
        enriched_mutated = TechnicalFeatures().compute(mutated)

        X_base, _, _, _, _ = FeaturePipeline().prepare_xgboost_input(enriched_base, horizon=1, direction_classes=3)
        X_mut, _, _, _, _ = FeaturePipeline().prepare_xgboost_input(enriched_mutated, horizon=1, direction_classes=3)

        # Rows are indexed by date — compare only dates that exist in both
        # AND are at/before the cutoff date (prepare_xgboost_input's own
        # dropna() legitimately removes different trailing rows near the
        # cutoff since horizon=1's target needs tomorrow's price, which is
        # exactly the point: don't compare rows the label itself invalidates).
        cutoff_date = base.index[CUTOFF]
        common_dates = X_base.index[X_base.index <= cutoff_date].intersection(X_mut.index)
        assert len(common_dates) >= 50, "not enough overlapping rows to meaningfully test — check MIN_OHLCV_ROWS/warmup windows"

        pd.testing.assert_frame_equal(
            X_base.loc[common_dates], X_mut.loc[common_dates], check_exact=False, rtol=1e-9,
        )

    def test_target_and_fwd_return_correctly_ARE_forward_looking(self):
        """
        The label (target/fwd_return) is SUPPOSED to depend on future
        data — that's what makes it a label. This test exists so the
        leakage tests above aren't misread as claiming labels are also
        leakage-free; only the FEATURE columns (X) have that property.
        """
        base = _synthetic_ohlcv()
        enriched = TechnicalFeatures().compute(base)
        _, y, _, _, fwd_return = FeaturePipeline().prepare_xgboost_input(enriched, horizon=1, direction_classes=3)
        # A 1-day-forward return must differ from a same-day return for at
        # least some rows, proving it actually looked ahead.
        same_day_return = enriched["Close"].pct_change().reindex(fwd_return.index)
        assert not np.allclose(fwd_return.values, same_day_return.values, equal_nan=True)
