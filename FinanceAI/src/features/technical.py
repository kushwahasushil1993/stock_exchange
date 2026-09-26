"""
Feature Engineering Pipeline:
  - Technical indicators (TA-Lib / pandas-ta)
  - Rolling statistics
  - Macro features
  - Fundamental ratios
  - News sentiment features
  - FNO-specific features (OI, PCR, IV)
"""
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Technical Indicators
# ════════════════════════════════════════════════════════════════════════════
class TechnicalFeatures:
    """
    Computes 40+ technical indicators covering:
    Trend | Momentum | Volatility | Volume | Pattern
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input:  OHLCV DataFrame with DatetimeIndex
        Output: DataFrame enriched with all technical features
        """
        df = df.copy().sort_index()
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df["Volume"]

        # ── Trend ────────────────────────────────────────────────────────────
        for w in [5, 10, 20, 50, 200]:
            df[f"sma_{w}"] = close.rolling(w).mean()
            df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()

        df["sma_20_slope"]  = df["sma_20"].diff(5) / 5
        df["sma_50_slope"]  = df["sma_50"].diff(5) / 5
        df["sma_200_slope"] = df["sma_200"].diff(5) / 5
        df["price_vs_sma50"]= (close - df["sma_50"]) / df["sma_50"]
        df["price_vs_ema20"]= (close - df["ema_20"]) / df["ema_20"]
        df["price_vs_sma200"]=(close - df["sma_200"]) / df["sma_200"]
        df["golden_cross"]  = (df["sma_50"] > df["sma_200"]).astype(int)

        # ── Swing structure (backward-looking only — see _swing_structure) ──
        swing = self._swing_structure(high, low, close)
        for col in swing.columns:
            df[col] = swing[col]

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"]         = ema12 - ema26
        df["macd_signal"]  = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]    = df["macd"] - df["macd_signal"]
        df["macd_crossover"] = (
            (df["macd"] > df["macd_signal"]) &
            (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        ).astype(int)
        df["macd_histogram_slope"] = df["macd_hist"].diff(5) / 5

        # Parabolic SAR (simplified)
        df["psar"] = self._parabolic_sar(high, low, close)

        # ── Momentum ─────────────────────────────────────────────────────────
        # RSI
        for period in [9, 14, 21]:
            df[f"rsi_{period}"] = self._rsi(close, period)
        df["rsi_slope"] = df["rsi_14"].diff(5) / 5
        df["rsi_divergence"] = self._rsi_divergence(close, df["rsi_14"])

        # Stochastic %K %D
        df["stoch_k"], df["stoch_d"] = self._stochastic(high, low, close)

        # Rate of Change
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = close.pct_change(period) * 100

        # Williams %R
        df["williams_r"] = self._williams_r(high, low, close, 14)

        # CCI (Commodity Channel Index)
        df["cci"] = self._cci(high, low, close, 20)

        # MFI (Money Flow Index)
        df["mfi"] = self._mfi(high, low, close, vol, 14)

        # ── Volatility ───────────────────────────────────────────────────────
        # Bollinger Bands
        df["bb_mid"]   = close.rolling(20).mean()
        bb_std         = close.rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std
        df["bb_pct_b"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

        # ATR (Average True Range)
        df["atr_14"] = self._atr(high, low, close, 14)
        df["atr_pct"] = df["atr_14"] / close

        # Historical Volatility
        log_ret = np.log(close / close.shift(1))
        for w in [10, 20, 30]:
            df[f"hv_{w}"] = log_ret.rolling(w).std() * np.sqrt(252) * 100

        # ── Volume ───────────────────────────────────────────────────────────
        df["vol_sma_20"]   = vol.rolling(20).mean()
        df["vol_ratio"]    = vol / df["vol_sma_20"]
        df["obv"]          = self._obv(close, vol)
        df["vwap"]         = self._vwap(high, low, close, vol)
        df["price_vs_vwap"]= (close - df["vwap"]) / df["vwap"]

        # Chaikin Money Flow
        df["cmf"] = self._cmf(high, low, close, vol, 20)

        # Force Index
        df["force_index"] = (close - close.shift(1)) * vol

        # ── Price Patterns ───────────────────────────────────────────────────
        df["daily_return"]  = close.pct_change() * 100
        df["weekly_return"] = close.pct_change(5) * 100
        df["monthly_return"]= close.pct_change(21) * 100
        df["gap_up"]        = ((df["Open"] > close.shift(1) * 1.01)).astype(int)
        df["gap_down"]      = ((df["Open"] < close.shift(1) * 0.99)).astype(int)
        df["high_52w"]      = high.rolling(252).max()
        df["low_52w"]       = low.rolling(252).min()
        df["pct_from_52w_high"] = (close - df["high_52w"]) / df["high_52w"] * 100

        # Candle features
        df["candle_body"]  = abs(df["Close"] - df["Open"]) / df["Open"] * 100
        df["upper_shadow"] = (high - df[["Open", "Close"]].max(axis=1)) / df["Open"] * 100
        df["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - low) / df["Open"] * 100

        # Trend strength
        df["adx"] = self._adx(high, low, close, 14)

        # ── Support / resistance distance ───────────────────────────────────
        # Rolling 20-bar high/low as a support/resistance proxy — both are
        # backward-looking only (pandas .rolling(20) at row i uses rows
        # [i-19, i], never i+1 onward), so this is safe to use as a
        # same-day feature for a next-session target.
        resistance = high.rolling(20).max()
        support = low.rolling(20).min()
        df["resistance_distance"] = (resistance - close) / close
        df["support_distance"] = (close - support) / close

        # ── Volume trend ─────────────────────────────────────────────────────
        df["volume_trend"] = df["vol_sma_20"].pct_change(10)

        return df.replace([np.inf, -np.inf], np.nan)

    # ── Helper Methods ───────────────────────────────────────────────────────
    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _stochastic(
        high: pd.Series, low: pd.Series, close: pd.Series,
        k_period: int = 14, d_period: int = 3
    ):
        lowest_low   = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        d = k.rolling(d_period).mean()
        return k, d

    @staticmethod
    def _williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        return -100 * (hh - close) / (hh - ll).replace(0, np.nan)

    @staticmethod
    def _cci(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
    ) -> pd.Series:
        tp = (high + low + close) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (tp - sma) / (0.015 * mad.replace(0, np.nan))

    @staticmethod
    def _mfi(
        high: pd.Series, low: pd.Series, close: pd.Series,
        vol: pd.Series, period: int = 14
    ) -> pd.Series:
        tp = (high + low + close) / 3
        raw_mf = tp * vol
        pos_mf = raw_mf.where(tp > tp.shift(1), 0).rolling(period).sum()
        neg_mf = raw_mf.where(tp < tp.shift(1), 0).rolling(period).sum()
        mfr = pos_mf / neg_mf.replace(0, np.nan)
        return 100 - 100 / (1 + mfr)

    @staticmethod
    def _atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _obv(close: pd.Series, vol: pd.Series) -> pd.Series:
        direction = np.sign(close.diff()).fillna(0)
        return (direction * vol).cumsum()

    @staticmethod
    def _vwap(
        high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series
    ) -> pd.Series:
        tp = (high + low + close) / 3
        return (tp * vol).cumsum() / vol.cumsum()

    @staticmethod
    def _cmf(
        high: pd.Series, low: pd.Series, close: pd.Series,
        vol: pd.Series, period: int = 20
    ) -> pd.Series:
        mfv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan) * vol
        return mfv.rolling(period).sum() / vol.rolling(period).sum()

    @staticmethod
    def _swing_structure(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10) -> pd.DataFrame:
        """
        higher_highs / higher_lows / lower_highs / lower_lows — whether this
        window's price extreme exceeds the PRIOR (non-overlapping) window's
        extreme. This is a backward-looking proxy for swing structure, not a
        literal fractal/pivot detector: a real pivot-point detector needs N
        bars AFTER a candidate point to confirm it's a local extreme, which
        would only be knowable in hindsight — using it as a same-day feature
        for a next-session target would be look-ahead leakage. Comparing two
        adjacent backward-only rolling windows avoids that at the cost of
        being a cruder signal.
        """
        roll_high = high.rolling(window).max()
        roll_low = low.rolling(window).min()
        prev_high = roll_high.shift(window)
        prev_low = roll_low.shift(window)
        return pd.DataFrame({
            "higher_highs": (roll_high > prev_high).astype(int),
            "lower_highs": (roll_high < prev_high).astype(int),
            "higher_lows": (roll_low > prev_low).astype(int),
            "lower_lows": (roll_low < prev_low).astype(int),
        }, index=high.index)

    @staticmethod
    def _rsi_divergence(close: pd.Series, rsi: pd.Series, window: int = 10) -> pd.Series:
        """
        +1 = bullish divergence proxy (price making a lower low over this
        window than the prior window, while RSI is making a higher low —
        weakening downward momentum), -1 = bearish divergence proxy (price
        higher high, RSI lower high), 0 = neither. Same backward-only,
        adjacent-rolling-window comparison as _swing_structure, for the same
        leakage-safety reason — not a strict pivot-to-pivot divergence
        detector.
        """
        price_hh = close.rolling(window).max() > close.rolling(window).max().shift(window)
        price_ll = close.rolling(window).min() < close.rolling(window).min().shift(window)
        rsi_lh = rsi.rolling(window).max() < rsi.rolling(window).max().shift(window)
        rsi_hl = rsi.rolling(window).min() > rsi.rolling(window).min().shift(window)
        bearish = price_hh & rsi_lh
        bullish = price_ll & rsi_hl
        return bullish.astype(int) - bearish.astype(int)

    @staticmethod
    def _adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        tr   = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        plus_dm  = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        # Smooth
        atr      = tr.rolling(period).mean()
        plus_di  = 100 * plus_dm.rolling(period).mean()  / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.rolling(period).mean()

    @staticmethod
    def _parabolic_sar(
        high: pd.Series, low: pd.Series, close: pd.Series,
        af_start: float = 0.02, af_max: float = 0.2
    ) -> pd.Series:
        """Simplified Parabolic SAR."""
        sar = close.copy()
        trend = 1  # 1 up, -1 down
        ep    = low.iloc[0]
        af    = af_start

        for i in range(2, len(close)):
            prev_sar = sar.iloc[i - 1]
            if trend == 1:
                sar.iloc[i] = prev_sar + af * (ep - prev_sar)
                sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2])
                if low.iloc[i] < sar.iloc[i]:
                    trend = -1
                    sar.iloc[i] = ep
                    ep = low.iloc[i]
                    af = af_start
                else:
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + af_start, af_max)
            else:
                sar.iloc[i] = prev_sar + af * (ep - prev_sar)
                sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2])
                if high.iloc[i] > sar.iloc[i]:
                    trend = 1
                    sar.iloc[i] = ep
                    ep = high.iloc[i]
                    af = af_start
                else:
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + af_start, af_max)
        return sar


# ════════════════════════════════════════════════════════════════════════════
# Feature Normalizer & Selector
# ════════════════════════════════════════════════════════════════════════════
class FeaturePipeline:
    """
    Combine technical, fundamental, macro, and sentiment features into
    a single normalized feature matrix ready for model ingestion.
    """

    LSTM_FEATURES = [
        "Close", "Open", "High", "Low", "Volume",
        "sma_20", "sma_50", "ema_20", "ema_50",
        "macd", "macd_signal", "macd_hist",
        "rsi_14", "rsi_9",
        "stoch_k", "stoch_d",
        "bb_pct_b", "bb_width",
        "atr_pct", "hv_20",
        "vol_ratio", "obv",
        "price_vs_vwap",
        "cmf", "force_index",
        "adx", "cci", "mfi",
        "daily_return", "weekly_return",
        "pct_from_52w_high",
    ]

    XGBOOST_FEATURES = LSTM_FEATURES + [
        "roc_5", "roc_10", "roc_20",
        # stoch_k is NOT re-listed here — it's already inherited from
        # LSTM_FEATURES above. Re-adding it used to produce two identically-
        # named "stoch_k" columns in the feature matrix, which both XGBoost
        # and LightGBM reject outright ("feature_names must be unique") —
        # meaning every attempt to actually train either model on this
        # feature set failed immediately, unconditionally, for every symbol.
        "williams_r",
        "gap_up", "gap_down",
        "candle_body", "upper_shadow", "lower_shadow",
        "golden_cross", "macd_crossover",
        # New technical features — swing structure, S/R distance, MA slopes,
        # RSI slope/divergence, MACD histogram slope, volume trend (all
        # backward-looking only — see TechnicalFeatures._swing_structure /
        # _rsi_divergence docstrings for the leakage-safety reasoning).
        "sma_50_slope", "sma_200_slope", "price_vs_ema20", "price_vs_sma200",
        "higher_highs", "lower_highs", "higher_lows", "lower_lows",
        "resistance_distance", "support_distance",
        "rsi_slope", "rsi_divergence", "macd_histogram_slope", "volume_trend",
        # Sector context (src/market/sector_index.py) — real NSE sectoral
        # index vs. this stock's own return, joined in by universe_scan.py/
        # train_three_models.py before this pipeline runs. NaN (not 0) for
        # a stock whose sector has no real index mapping (e.g. Utilities) —
        # tree models handle NaN as a legitimate "missing" split natively.
        "sector_return", "return_vs_sector", "rolling_beta_sector", "rolling_corr_sector",
        # Fundamental features added later
        "pe_ratio", "pb_ratio", "roe", "debt_to_equity",
        "revenue_growth_yoy", "profit_margin", "beta",
        # Sentiment feature added later
        "sentiment_score",
        # Macro features added later
        "usd_inr_ret", "gold_ret", "crude_ret", "vix",
    ]

    def prepare_lstm_input(
        self,
        df: pd.DataFrame,
        lookback: int = 60,
        target_col: str = "Close",
    ) -> tuple:
        """
        Returns (X, y) where:
          X shape = (samples, lookback, features)
          y shape = (samples,)  — next-day return classification
        """
        from sklearn.preprocessing import StandardScaler

        features = [f for f in self.LSTM_FEATURES if f in df.columns]
        data = df[features].dropna()

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        X, y = [], []
        for i in range(lookback, len(scaled) - 1):
            X.append(scaled[i - lookback:i])
            # Binary classification: 1 if next-day close > today
            ret = (data[target_col].iloc[i + 1] - data[target_col].iloc[i]) / data[target_col].iloc[i]
            y.append(1 if ret > 0 else 0)

        return np.array(X), np.array(y), scaler, features

    def prepare_xgboost_input(
        self,
        df: pd.DataFrame,
        fundamental: Optional[Dict] = None,
        sentiment: Optional[float] = None,
        macro: Optional[Dict] = None,
        horizon: int = 5,
        direction_classes: int = 2,
        flat_threshold: float = 0.002,
        return_threshold: float = 0.02,
    ) -> tuple:
        """
        Prepare tabular features for XGBoost/LightGBM, plus BOTH a
        classification target and the continuous forward return in one
        pass (so a direction model and a return-regression model can be
        trained from a single feature computation instead of duplicating
        it) — see Model A/Model B in src/models/direction_model.py and
        src/models/return_model.py.

        direction_classes=2 (default, backward-compatible with existing
        callers): target = 1 if forward return > return_threshold, else 0.
        direction_classes=3: target = 2 (UP) if forward return >
        flat_threshold, 0 (DOWN) if < -flat_threshold, else 1 (FLAT) — a
        small dead-zone around zero so noise-level moves aren't forced into
        a direction, matching "estimate probabilities... not force a
        prediction."

        Returns (X, y, scaler, features, fwd_return) — fwd_return is the
        raw, unthresholded forward return for the SAME rows as X/y, for
        regression training.
        """
        data = df.copy()

        # Target first, dropna on technical + target ONLY, before injecting
        # the scalar fundamental/macro columns below. Those scalars broadcast
        # one value across every historical row for this symbol — if any
        # single one is missing (common: Yahoo has no "beta" for plenty of
        # small/mid-caps) and dropna() ran across the combined frame, that
        # one all-NaN column would silently wipe out EVERY row for the whole
        # symbol, collapsing X to 0 rows. That used to happen invisibly:
        # callers just saw an empty X, treated it as "no data", and fell
        # through to the neutral 0.5 placeholder — for any symbol missing
        # even one fundamental field, not just ones truly short on history.
        data["fwd_return"] = data["Close"].pct_change(horizon).shift(-horizon)
        if direction_classes == 3:
            data["target"] = np.select(
                [data["fwd_return"] > flat_threshold, data["fwd_return"] < -flat_threshold],
                [2, 0],  # 2=UP, 0=DOWN
                default=1,  # 1=FLAT
            )
        else:
            data["target"] = (data["fwd_return"] > return_threshold).astype(int)
        tech_features = [f for f in self.XGBOOST_FEATURES if f in data.columns]
        data = data[tech_features + ["target", "fwd_return"]].dropna()

        # Inject scalar features — always as a properly-typed float column,
        # for every field XGBOOST_FEATURES expects, even when this symbol
        # has no value for it (np.nan, not left absent, and not the bare
        # Python `None`: `data[k] = None` makes pandas infer an object-dtype
        # column, which LightGBM rejects outright — "pandas dtypes must be
        # int, float or bool" — and skipping the column entirely would shift
        # every later feature out of position for XGBoost's positional
        # (.values) inference path). Emitting the same full column set on
        # every call, training and inference alike, keeps every row's
        # feature alignment identical regardless of which scalar values
        # happen to be available for a given symbol. XGBoost/LightGBM's
        # hist-based tree building treats NaN as a legitimate "missing"
        # split value natively, so no imputation is needed either.
        scalar_sources: Dict[str, Any] = {}
        if fundamental:
            scalar_sources.update(fundamental)
        if sentiment is not None:
            scalar_sources["sentiment_score"] = sentiment
        if macro:
            scalar_sources.update(macro)

        SCALAR_FEATURES = [
            "pe_ratio", "pb_ratio", "roe", "debt_to_equity",
            "revenue_growth_yoy", "profit_margin", "beta",
            "sentiment_score",
            "usd_inr_ret", "gold_ret", "crude_ret", "vix",
        ]
        for k in SCALAR_FEATURES:
            if k not in self.XGBOOST_FEATURES or k in data.columns:
                continue
            v = scalar_sources.get(k)
            try:
                data[k] = float(v) if v is not None else np.nan
            except (TypeError, ValueError):
                data[k] = np.nan

        features = [f for f in self.XGBOOST_FEATURES if f in data.columns]
        X = data[features]
        y = data["target"]
        fwd_return = data["fwd_return"]

        # Tree models don't need feature scaling (invariant to any monotonic
        # per-feature transform) — returning raw X keeps train and inference
        # using the exact same untransformed values, and avoids re-fitting a
        # StandardScaler on whatever single row is being scored at inference
        # time (which the old code did, and which especially breaks for a
        # single-row inference call since a 1-row fit has zero variance).
        scaler = None
        X_scaled = X

        return X_scaled, y, scaler, features, fwd_return
