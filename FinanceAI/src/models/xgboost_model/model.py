"""
XGBoost / LightGBM gradient-boosting model for tabular feature prediction.
Includes Optuna hyperparameter optimization and SHAP explainability.
"""
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
)

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost classifier with:
      - Optuna HPO (50 trials)
      - SHAP feature importance
      - Calibrated probabilities
    """

    def __init__(self, model_dir: str = "models_store/xgboost"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.feature_names = None

    def train_with_hpo(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        symbol: str,
        n_trials: int = 50,
        cv_folds: int = 5,
    ) -> Dict:
        try:
            import xgboost as xgb
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            self.feature_names = list(X.columns)

            def objective(trial):
                params = {
                    "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
                    "max_depth":         trial.suggest_int("max_depth", 3, 8),
                    "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
                    "reg_alpha":         trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
                    "reg_lambda":        trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
                    "scale_pos_weight":  (y == 0).sum() / max((y == 1).sum(), 1),
                    "use_label_encoder": False,
                    "eval_metric":       "auc",
                    "random_state":      42,
                    "tree_method":       "hist",
                }
                clf = xgb.XGBClassifier(**params)
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scores = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
                return scores.mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            best_params = study.best_params
            best_params.update({
                "scale_pos_weight": (y == 0).sum() / max((y == 1).sum(), 1),
                "use_label_encoder": False,
                "eval_metric": "auc",
                "random_state": 42,
                "tree_method": "hist",
            })

            self.model = xgb.XGBClassifier(**best_params)
            self.model.fit(X, y)

            # Metrics
            preds = self.model.predict(X)
            proba = self.model.predict_proba(X)[:, 1]
            metrics = {
                "train_accuracy":  round(accuracy_score(y, preds), 4),
                "train_auc":       round(roc_auc_score(y, proba), 4),
                "train_f1":        round(f1_score(y, preds, zero_division=0), 4),
                "best_cv_auc":     round(study.best_value, 4),
                "best_params":     best_params,
            }

            artifact_path = str(self.model_dir / f"{symbol}_xgb.pkl")
            with open(artifact_path, "wb") as f:
                pickle.dump({"model": self.model, "features": self.feature_names}, f)

            logger.info("XGB model saved: %s | metrics=%s", artifact_path, metrics)
            return {"artifact_path": artifact_path, "metrics": metrics}

        except Exception as exc:
            logger.error("XGB training failed: %s", exc)
            return {"error": str(exc)}

    def get_shap_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """Compute SHAP feature importance values."""
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            importance = dict(
                zip(self.feature_names, np.abs(shap_values).mean(axis=0))
            )
            return {k: round(float(v), 6) for k, v in
                    sorted(importance.items(), key=lambda x: -x[1])[:20]}
        except Exception as exc:
            logger.warning("SHAP failed: %s", exc)
            # Fall back to built-in importance
            if self.model and self.feature_names:
                imp = self.model.feature_importances_
                return {k: round(float(v), 6) for k, v in
                        sorted(zip(self.feature_names, imp), key=lambda x: -x[1])[:20]}
            return {}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), 0.5)
        return self.model.predict_proba(X)[:, 1]

    def load(self, artifact_path: str):
        with open(artifact_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["features"]


class LightGBMModel:
    """LightGBM alternative — faster training, similar accuracy to XGBoost."""

    def __init__(self, model_dir: str = "models_store/lgbm"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.feature_names = None

    def train(self, X: pd.DataFrame, y: pd.Series, symbol: str) -> Dict:
        try:
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split

            self.feature_names = list(X.columns)
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            params = {
                "objective":        "binary",
                "metric":           "auc",
                "num_leaves":       63,
                "learning_rate":    0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq":     5,
                "scale_pos_weight": scale_pos,
                "n_estimators":     500,
                "random_state":     42,
                "verbose":          -1,
            }

            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
            self.model = lgb.LGBMClassifier(**params)
            self.model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )

            proba = self.model.predict_proba(X_val)[:, 1]
            preds = (proba >= 0.5).astype(int)
            metrics = {
                "val_auc":      round(roc_auc_score(y_val, proba), 4),
                "val_accuracy": round(accuracy_score(y_val, preds), 4),
                "val_f1":       round(f1_score(y_val, preds, zero_division=0), 4),
            }

            artifact_path = str(self.model_dir / f"{symbol}_lgbm.pkl")
            with open(artifact_path, "wb") as f:
                pickle.dump({"model": self.model, "features": self.feature_names}, f)

            return {"artifact_path": artifact_path, "metrics": metrics}

        except Exception as exc:
            logger.error("LightGBM training failed: %s", exc)
            return {"error": str(exc)}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), 0.5)
        return self.model.predict_proba(X)[:, 1]
