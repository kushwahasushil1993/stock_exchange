"""
The three-model prediction path (§20-21) — Direction + Expected Return +
Risk/Liquidity, combined by the transparent decision engine. This is
ADDITIVE alongside the existing ModelOrchestrator (src/models/ensemble/
predictor.py), not a replacement of it: if the walk-forward-trained
artifacts (src/pipelines/train_three_models.py) don't exist yet, callers
should fall back to the legacy path rather than crash — same "reuse good
existing components" principle the rest of this pass has followed.
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.database.models import DecisionType, LiquidityLevel, MarketRegime, RiskLevel

logger = logging.getLogger("three_model_orchestrator")

MODEL_DIR = Path("models_store/three_model")
HORIZON_DAYS = 1


def artifacts_exist() -> bool:
    return (MODEL_DIR / "direction_model.pkl").exists() and (MODEL_DIR / "return_model.pkl").exists()


class ThreeModelOrchestrator:
    def __init__(self):
        self._direction = None
        self._return = None

    def _load(self):
        if self._direction is None:
            with open(MODEL_DIR / "direction_model.pkl", "rb") as f:
                self._direction = pickle.load(f)
        if self._return is None:
            with open(MODEL_DIR / "return_model.pkl", "rb") as f:
                self._return = pickle.load(f)

    def predict(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        nifty_df: Optional[pd.DataFrame] = None,
        sensex_df: Optional[pd.DataFrame] = None,
        sector_df: Optional[pd.DataFrame] = None,
        market_regime: Optional[MarketRegime] = None,
    ) -> Dict:
        """
        Returns a dict matching the new Prediction columns directly
        (probability_up/down/flat, expected_return(+bounds), risk_level,
        liquidity_level, decision, decision_factors) plus current_price/
        target_price/stop_loss for backward-compat display. Raises if the
        artifacts aren't trained yet or features can't be computed — the
        caller (universe_scan.py) is expected to catch this and fall back
        to the legacy ModelOrchestrator, exactly like every other
        best-effort fallback in this codebase.
        """
        from src.features.technical import TechnicalFeatures, FeaturePipeline
        from src.market.relative_strength import add_index_relative_features
        from src.market.risk_liquidity import assess_risk_liquidity
        from src.market.decision_engine import decide

        self._load()

        enriched = TechnicalFeatures().compute(ohlcv_df)
        enriched = add_index_relative_features(enriched, nifty_df, "nifty")
        enriched = add_index_relative_features(enriched, sensex_df, "sensex")
        # sector_df is None whenever this stock's sector has no real NSE
        # index mapping (see src/market/sector_index.py) or hasn't been
        # scanned yet — add_index_relative_features no-ops safely on None.
        enriched = add_index_relative_features(enriched, sector_df, "sector")

        fp = FeaturePipeline()
        X, _y, _scaler, _features, _fwd = fp.prepare_xgboost_input(
            enriched, fundamental=None, sentiment=None, macro=None,
            horizon=HORIZON_DAYS, direction_classes=3,
        )
        if X.empty:
            raise ValueError(f"{symbol}: insufficient data to build three-model features")

        dir_model = self._direction["model"]
        dir_features = self._direction["features"]
        row = X.iloc[-1:].reindex(columns=dir_features, fill_value=np.nan)
        proba = dir_model.predict_proba(row)[0]  # [DOWN, FLAT, UP] per class order 0,1,2
        prob_down, prob_flat, prob_up = float(proba[0]), float(proba[1]), float(proba[2])

        ret_features = self._return["features"]
        ret_row = X.iloc[-1:].reindex(columns=ret_features, fill_value=np.nan)
        expected_return = float(self._return["mid"].predict(ret_row)[0])
        expected_return_lower = float(self._return["lo"].predict(ret_row)[0])
        expected_return_upper = float(self._return["hi"].predict(ret_row)[0])

        risk_result = assess_risk_liquidity(enriched)

        decision = decide(
            probability_up=prob_up, probability_down=prob_down, probability_flat=prob_flat,
            expected_return=expected_return, risk_level=risk_result.risk_level,
            liquidity_level=risk_result.liquidity_level, market_regime=market_regime,
        )

        last_close = float(enriched["Close"].iloc[-1])
        atr = float(enriched["atr_14"].iloc[-1]) if "atr_14" in enriched and pd.notna(enriched["atr_14"].iloc[-1]) else last_close * 0.02
        target_price = last_close * (1 + expected_return)
        stop_loss = last_close - 1.5 * atr

        return {
            "symbol": symbol,
            "current_price": round(last_close, 2),
            "probability_up": round(prob_up, 4), "probability_down": round(prob_down, 4), "probability_flat": round(prob_flat, 4),
            "expected_return": round(expected_return, 5), "expected_return_lower": round(expected_return_lower, 5),
            "expected_return_upper": round(expected_return_upper, 5),
            "risk_level": risk_result.risk_level, "liquidity_level": risk_result.liquidity_level,
            "risk_liquidity_details": risk_result.details,
            "market_regime": market_regime,
            "decision": decision.decision,
            "decision_factors": {"positive": decision.positive_factors, "negative": decision.negative_factors},
            "target_price": round(target_price, 2), "stop_loss": round(stop_loss, 2),
            "model_type": "three_model_v2", "feature_version": self._direction.get("feature_version"),
            "predicted_at": pd.Timestamp.utcnow().isoformat(),
        }
