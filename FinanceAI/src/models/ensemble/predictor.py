"""
Ensemble model: weighted combination of LSTM + XGBoost + LightGBM.
Uses meta-learner (Logistic Regression) trained on OOF predictions — stacking.
"""
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Level-2 stacking ensemble.
    Base models: LSTM (time-series), XGBoost (tabular), LightGBM (tabular)
    Meta-learner: Logistic Regression with calibrated probabilities.
    """

    MODEL_WEIGHTS = {
        "lstm":    0.35,
        "xgboost": 0.40,
        "lgbm":    0.25,
    }

    def __init__(self, model_dir: str = "models_store/ensemble"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.meta_learner: Optional[LogisticRegression] = None

    # ── Weighted average ensemble ────────────────────────────────────────────
    def predict_weighted(
        self,
        lstm_prob: float,
        xgb_prob: float,
        lgbm_prob: float,
    ) -> float:
        """Weighted average — no meta-learner needed."""
        w = self.MODEL_WEIGHTS
        return (
            w["lstm"]    * lstm_prob +
            w["xgboost"] * xgb_prob +
            w["lgbm"]    * lgbm_prob
        )

    # ── Stacking (train meta-learner on OOF predictions) ────────────────────
    def train_meta_learner(
        self,
        oof_lstm:   np.ndarray,
        oof_xgb:    np.ndarray,
        oof_lgbm:   np.ndarray,
        y_true:     np.ndarray,
        symbol:     str,
    ) -> Dict:
        """Train Logistic Regression on OOF base-model predictions."""
        X_meta = np.column_stack([oof_lstm, oof_xgb, oof_lgbm])
        self.meta_learner = CalibratedClassifierCV(
            LogisticRegression(C=1.0, random_state=42),
            cv=5, method="isotonic",
        )
        self.meta_learner.fit(X_meta, y_true)

        from sklearn.metrics import roc_auc_score
        proba = self.meta_learner.predict_proba(X_meta)[:, 1]
        metrics = {"meta_auc": round(roc_auc_score(y_true, proba), 4)}

        artifact_path = str(self.model_dir / f"{symbol}_ensemble.pkl")
        with open(artifact_path, "wb") as f:
            pickle.dump(self.meta_learner, f)

        logger.info("Ensemble meta-learner saved: %s", artifact_path)
        return {"artifact_path": artifact_path, "metrics": metrics}

    def predict_stacked(
        self,
        lstm_prob: float,
        xgb_prob: float,
        lgbm_prob: float,
    ) -> float:
        if self.meta_learner is None:
            return self.predict_weighted(lstm_prob, xgb_prob, lgbm_prob)
        X = np.array([[lstm_prob, xgb_prob, lgbm_prob]])
        return float(self.meta_learner.predict_proba(X)[0, 1])

    def load(self, artifact_path: str):
        with open(artifact_path, "rb") as f:
            self.meta_learner = pickle.load(f)


class ModelOrchestrator:
    """
    Orchestrates the full prediction pipeline for a single symbol:
    1. Fetch & feature-engineer data
    2. Load all base models
    3. Run ensemble prediction
    4. Compute composite score (technical + fundamental + sentiment)
    5. Output trading signal with confidence, target price, stop-loss
    """

    def __init__(self, model_base_dir: str = "models_store"):
        self.model_base = Path(model_base_dir)

    def predict(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        fundamental: Optional[Dict] = None,
        sentiment_score: float = 0.0,
        macro: Optional[Dict] = None,
    ) -> Dict:
        """
        Returns full prediction result dict.
        """
        from src.features.technical import TechnicalFeatures, FeaturePipeline
        from src.models.lstm.model import LSTMTrainer
        from src.models.xgboost_model.model import XGBoostModel, LightGBMModel
        from config.settings import settings

        tf_eng  = TechnicalFeatures()
        fp      = FeaturePipeline()
        symbol_clean = symbol.replace(".NS", "").replace("^", "")

        # ── Feature engineering ──────────────────────────────────────────────
        enriched = tf_eng.compute(ohlcv_df)

        # ── LSTM prediction ──────────────────────────────────────────────────
        lstm_prob = 0.5
        lstm_path = self.model_base / "lstm" / f"{symbol_clean}_lstm.keras"
        if lstm_path.exists():
            X_seq, _, scaler, _ = fp.prepare_lstm_input(enriched, lookback=settings.lstm_lookback)
            if len(X_seq) > 0:
                trainer   = LSTMTrainer()
                preds     = trainer.predict(str(lstm_path), X_seq[-1:])
                lstm_prob = float(preds[0])

        # ── XGBoost prediction ────────────────────────────────────────────────
        xgb_prob = 0.5
        xgb_path = self.model_base / "xgboost" / f"{symbol_clean}_xgb.pkl"
        if xgb_path.exists():
            X_tab, _, _, _ = fp.prepare_xgboost_input(enriched, fundamental, sentiment_score, macro)
            if len(X_tab) > 0:
                xgb = XGBoostModel()
                xgb.load(str(xgb_path))
                xgb_prob = float(xgb.predict_proba(X_tab.iloc[-1:].values)[0])

        # ── LightGBM prediction ───────────────────────────────────────────────
        lgbm_prob = 0.5
        lgbm_path = self.model_base / "lgbm" / f"{symbol_clean}_lgbm.pkl"
        if lgbm_path.exists():
            X_tab2, _, _, _ = fp.prepare_xgboost_input(enriched, fundamental, sentiment_score, macro)
            if len(X_tab2) > 0:
                lgbm = LightGBMModel()
                with open(lgbm_path, "rb") as fh:
                    data = pickle.load(fh)
                    lgbm.model = data["model"]
                    lgbm.feature_names = data["features"]
                lgbm_prob = float(lgbm.predict_proba(X_tab2.iloc[-1:])[0])

        # ── Ensemble ─────────────────────────────────────────────────────────
        ensemble = EnsemblePredictor()
        ensemble_path = self.model_base / "ensemble" / f"{symbol_clean}_ensemble.pkl"
        if ensemble_path.exists():
            ensemble.load(str(ensemble_path))
        final_prob = ensemble.predict_stacked(lstm_prob, xgb_prob, lgbm_prob)

        # ── Composite score ───────────────────────────────────────────────────
        tech_score = self._technical_score(enriched)
        fund_score = self._fundamental_score(fundamental or {})
        comp_score = 0.45 * final_prob + 0.35 * tech_score + 0.20 * fund_score

        # ── Signal ────────────────────────────────────────────────────────────
        if comp_score >= 0.65:
            signal = "BUY"
        elif comp_score <= 0.35:
            signal = "SELL"
        else:
            signal = "HOLD"

        # ── Target price & stop-loss (ATR-based) ──────────────────────────────
        last_close = float(enriched["Close"].iloc[-1])
        atr        = float(enriched["atr_14"].iloc[-1]) if "atr_14" in enriched else last_close * 0.02
        target     = last_close + 2.5 * atr
        stop_loss  = last_close - 1.5 * atr
        rr         = round((target - last_close) / (last_close - stop_loss + 1e-6), 2)

        # ── SHAP feature importance ───────────────────────────────────────────
        shap_imp = {}
        if xgb_path.exists():
            xgb2 = XGBoostModel()
            xgb2.load(str(xgb_path))
            X_tab3, _, _, _ = fp.prepare_xgboost_input(enriched, fundamental, sentiment_score, macro)
            shap_imp = xgb2.get_shap_importance(X_tab3.iloc[-10:])

        return {
            "symbol":            symbol,
            "signal":            signal,
            "confidence":        round(comp_score, 4),
            "lstm_prob":         round(lstm_prob, 4),
            "xgb_prob":          round(xgb_prob, 4),
            "lgbm_prob":         round(lgbm_prob, 4),
            "ensemble_prob":     round(final_prob, 4),
            "technical_score":   round(tech_score, 4),
            "fundamental_score": round(fund_score, 4),
            "composite_score":   round(comp_score, 4),
            "sentiment_score":   round(sentiment_score, 4),
            "current_price":     round(last_close, 2),
            "target_price":      round(target, 2),
            "stop_loss":         round(stop_loss, 2),
            "risk_reward_ratio": rr,
            "predicted_return_5d": round((target - last_close) / last_close * 100, 2),
            "feature_importance": shap_imp,
            "predicted_at":      pd.Timestamp.utcnow().isoformat(),
        }

    def _technical_score(self, df: pd.DataFrame) -> float:
        """Score 0–1 based on RSI, MACD, BB position, ADX."""
        score = 0.5
        try:
            row = df.iloc[-1]
            pts, total = 0.0, 0.0

            if "rsi_14" in row and pd.notna(row["rsi_14"]):
                total += 1
                if 40 < row["rsi_14"] < 70:
                    pts += 0.8
                elif row["rsi_14"] <= 30:
                    pts += 1.0  # Oversold = buying opportunity
                elif row["rsi_14"] >= 70:
                    pts += 0.3

            if "macd_hist" in row and pd.notna(row["macd_hist"]):
                total += 1
                pts += (1.0 if row["macd_hist"] > 0 else 0.0)

            if "bb_pct_b" in row and pd.notna(row["bb_pct_b"]):
                total += 1
                b = row["bb_pct_b"]
                pts += (1 - abs(b - 0.5) / 0.5) * 0.8 if 0 < b < 1 else 0.2

            if "adx" in row and pd.notna(row["adx"]):
                total += 1
                pts += (1.0 if row["adx"] > 25 else 0.5)

            if "golden_cross" in row:
                total += 1
                pts += row["golden_cross"]

            score = pts / total if total > 0 else 0.5
        except Exception:
            pass
        return float(np.clip(score, 0.0, 1.0))

    def _fundamental_score(self, fundamental: Dict) -> float:
        """Score 0–1 based on PE, ROE, D/E, revenue growth."""
        score = 0.5
        pts, total = 0.0, 0.0
        try:
            pe = fundamental.get("pe_ratio")
            if pe and pe > 0:
                total += 1
                pts += 1.0 if pe < 20 else (0.7 if pe < 35 else 0.3)

            roe = fundamental.get("roe")
            if roe is not None:
                total += 1
                pts += 1.0 if roe > 0.20 else (0.7 if roe > 0.12 else 0.3)

            de = fundamental.get("debt_to_equity")
            if de is not None:
                total += 1
                pts += 1.0 if de < 0.5 else (0.6 if de < 1.5 else 0.2)

            rev = fundamental.get("revenue_growth_yoy")
            if rev is not None:
                total += 1
                pts += 1.0 if rev > 0.20 else (0.7 if rev > 0.10 else 0.3)

            score = pts / total if total > 0 else 0.5
        except Exception:
            pass
        return float(np.clip(score, 0.0, 1.0))
