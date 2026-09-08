"""
LSTM Model for time-series stock prediction.
Architecture: Stacked Bi-LSTM → Attention → Dense
"""
import os
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class AttentionLayer:
    """Bahdanau-style attention — implemented as a Keras layer."""
    pass  # defined inline below via tf.keras


def build_lstm_model(
    input_shape: Tuple[int, int],
    lstm_units: int = 128,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
):
    """
    Stacked Bi-LSTM with self-attention for sequence classification.

    input_shape: (lookback_steps, n_features)
    Output:      sigmoid probability (BUY signal)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model, Input

        inp = Input(shape=input_shape, name="ohlcv_sequence")

        # Bi-LSTM stack
        x = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, recurrent_dropout=0.1),
            name="bilstm_1",
        )(inp)
        x = layers.Dropout(dropout_rate)(x)

        x = layers.Bidirectional(
            layers.LSTM(lstm_units // 2, return_sequences=True, recurrent_dropout=0.1),
            name="bilstm_2",
        )(x)
        x = layers.Dropout(dropout_rate)(x)

        # Self-Attention
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=lstm_units // 4)(x, x)
        x    = layers.LayerNormalization()(x + attn)

        # Global context
        x = layers.GlobalAveragePooling1D()(x)

        # Classification head
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid", name="buy_prob")(x)

        model = Model(inputs=inp, outputs=out, name="FinAI_LSTM")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        return model

    except ImportError:
        logger.warning("TensorFlow not installed. LSTM model unavailable.")
        return None


class LSTMTrainer:
    """Trains, evaluates, and persists the LSTM model."""

    def __init__(self, model_dir: str = "models_store/lstm"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        symbol: str,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> Dict:
        try:
            import tensorflow as tf

            input_shape = (X_train.shape[1], X_train.shape[2])
            model = build_lstm_model(input_shape)
            if model is None:
                return {"error": "TF not available"}

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc", patience=10, restore_best_weights=True, mode="max"
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    self.model_dir / f"{symbol}_best.keras",
                    save_best_only=True,
                    monitor="val_auc",
                    mode="max",
                ),
            ]

            # Class weight for imbalanced labels
            pos = y_train.sum()
            neg = len(y_train) - pos
            class_weight = {0: 1.0, 1: neg / pos if pos > 0 else 1.0}

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=0,
            )

            # Evaluate
            val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
            metrics = {
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
                "val_auc": round(val_auc, 4),
                "epochs_trained": len(history.history["loss"]),
            }

            # Save final model
            artifact_path = str(self.model_dir / f"{symbol}_lstm.keras")
            model.save(artifact_path)
            logger.info("LSTM model saved to %s | metrics=%s", artifact_path, metrics)

            return {"artifact_path": artifact_path, "metrics": metrics}

        except Exception as exc:
            logger.error("LSTM training failed: %s", exc)
            return {"error": str(exc)}

    def predict(self, model_path: str, X: np.ndarray) -> np.ndarray:
        """Load saved model and return probabilities."""
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            return model.predict(X, verbose=0).flatten()
        except Exception as exc:
            logger.error("LSTM predict error: %s", exc)
            return np.full(len(X), 0.5)
