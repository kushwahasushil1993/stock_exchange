"""
Unit tests for feature engineering and recommendation engine.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.technical import TechnicalFeatures, FeaturePipeline
from src.recommendation.engine import (
    StockScreener, MutualFundRecommender, FNOAnalyzer, RiskProfile, TradingAgents
)


# ─── Fixtures ─────────────────────────────────────────────────────────────
@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """500 days of synthetic OHLCV data."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range(end=datetime.today(), periods=n, freq="B")
    close = 1000 + np.cumsum(np.random.randn(n) * 15)
    close = np.maximum(close, 100)
    df = pd.DataFrame({
        "Open":   close * (1 + np.random.randn(n) * 0.005),
        "High":   close * (1 + np.abs(np.random.randn(n)) * 0.01),
        "Low":    close * (1 - np.abs(np.random.randn(n)) * 0.01),
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)
    return df


@pytest.fixture
def sample_predictions() -> list:
    return [
        {
            "symbol": "RELIANCE.NS", "signal": "BUY",  "confidence": 0.75,
            "composite_score": 0.72, "technical_score": 0.68, "fundamental_score": 0.70,
            "sentiment_score": 0.30, "current_price": 2900, "target_price": 3100,
            "stop_loss": 2800, "risk_reward_ratio": 2.0, "predicted_return_5d": 6.9,
            "ensemble_prob": 0.74, "feature_importance": {}
        },
        {
            "symbol": "TCS.NS", "signal": "BUY", "confidence": 0.80,
            "composite_score": 0.78, "technical_score": 0.75, "fundamental_score": 0.80,
            "sentiment_score": 0.50, "current_price": 3900, "target_price": 4200,
            "stop_loss": 3750, "risk_reward_ratio": 2.0, "predicted_return_5d": 7.7,
            "ensemble_prob": 0.80, "feature_importance": {}
        },
        {
            "symbol": "INFY.NS", "signal": "SELL", "confidence": 0.65,
            "composite_score": 0.35, "technical_score": 0.30, "fundamental_score": 0.40,
            "sentiment_score": -0.20, "current_price": 1500, "target_price": 1350,
            "stop_loss": 1560, "risk_reward_ratio": 2.5, "predicted_return_5d": -10.0,
            "ensemble_prob": 0.28, "feature_importance": {}
        },
        {
            "symbol": "WIPRO.NS", "signal": "HOLD", "confidence": 0.45,
            "composite_score": 0.50, "technical_score": 0.50, "fundamental_score": 0.50,
            "sentiment_score": 0.0, "current_price": 480, "target_price": 500,
            "stop_loss": 465, "risk_reward_ratio": 1.3, "predicted_return_5d": 4.0,
            "ensemble_prob": 0.50, "feature_importance": {}
        },
    ]


@pytest.fixture
def sample_mf_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "scheme_code": "120503", "scheme_name": "Axis Bluechip Fund",
            "category": "Large Cap", "amc": "Axis AMC",
            "nav": 52.5, "return_1y": 18.0, "return_3y": 16.5, "return_5y": 15.2,
            "sharpe_ratio": 1.8, "sortino_ratio": 2.1, "alpha": 4.5,
            "expense_ratio": 0.65, "aum": 35000,
        },
        {
            "scheme_code": "100016", "scheme_name": "Mirae Asset Large Cap Fund",
            "category": "Large Cap", "amc": "Mirae",
            "nav": 98.3, "return_1y": 20.0, "return_3y": 18.0, "return_5y": 16.8,
            "sharpe_ratio": 2.1, "sortino_ratio": 2.5, "alpha": 5.2,
            "expense_ratio": 0.55, "aum": 32000,
        },
        {
            "scheme_code": "118825", "scheme_name": "Parag Parikh Flexi Cap Fund",
            "category": "Flexi Cap", "amc": "PPFAS",
            "nav": 68.2, "return_1y": 22.0, "return_3y": 19.5, "return_5y": 17.5,
            "sharpe_ratio": 2.3, "sortino_ratio": 2.8, "alpha": 6.0,
            "expense_ratio": 0.72, "aum": 55000,
        },
    ])


# ─── Technical Features Tests ────────────────────────────────────────────
class TestTechnicalFeatures:

    def test_compute_returns_columns(self, sample_ohlcv):
        tf = TechnicalFeatures()
        enriched = tf.compute(sample_ohlcv)
        for col in ["rsi_14", "macd", "bb_pct_b", "atr_14", "adx", "obv"]:
            assert col in enriched.columns, f"Missing column: {col}"

    def test_rsi_bounds(self, sample_ohlcv):
        tf = TechnicalFeatures()
        enriched = tf.compute(sample_ohlcv)
        rsi = enriched["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI out of [0, 100]"

    def test_no_inf_values(self, sample_ohlcv):
        tf = TechnicalFeatures()
        enriched = tf.compute(sample_ohlcv)
        assert not np.isinf(enriched.select_dtypes(include=[np.number])).any().any()

    def test_bollinger_bands_order(self, sample_ohlcv):
        tf = TechnicalFeatures()
        enriched = tf.compute(sample_ohlcv).dropna()
        assert (enriched["bb_upper"] > enriched["bb_lower"]).all()

    def test_feature_pipeline_lstm_shape(self, sample_ohlcv):
        tf = TechnicalFeatures()
        fp = FeaturePipeline()
        enriched = tf.compute(sample_ohlcv)
        X, y, scaler, feats = fp.prepare_lstm_input(enriched, lookback=30)
        assert X.ndim == 3
        assert X.shape[1] == 30
        assert len(X) == len(y)

    def test_feature_pipeline_xgboost_shape(self, sample_ohlcv):
        tf = TechnicalFeatures()
        fp = FeaturePipeline()
        enriched = tf.compute(sample_ohlcv)
        X, y, scaler, feats = fp.prepare_xgboost_input(enriched)
        assert X.shape[0] == len(y)
        assert X.shape[1] > 0


# ─── Stock Screener Tests ────────────────────────────────────────────────
class TestStockScreener:

    def test_buy_filter(self, sample_predictions):
        screener = StockScreener()
        buys = screener.screen(sample_predictions, top_n=5, signal_filter="BUY")
        assert all(r.signal == "BUY" for r in buys)

    def test_min_confidence_filter(self, sample_predictions):
        screener = StockScreener()
        results  = screener.screen(sample_predictions)
        assert all(r.confidence >= screener.MIN_CONFIDENCE for r in results)

    def test_min_rr_filter(self, sample_predictions):
        screener = StockScreener()
        results  = screener.screen(sample_predictions)
        assert all(r.risk_reward_ratio >= screener.MIN_RR_RATIO for r in results)

    def test_ranking_order(self, sample_predictions):
        screener = StockScreener()
        results  = screener.screen(sample_predictions)
        scores   = [r.composite_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_limit(self, sample_predictions):
        screener = StockScreener()
        results  = screener.screen(sample_predictions, top_n=1)
        assert len(results) <= 1


# ─── Mutual Fund Recommender Tests ──────────────────────────────────────
class TestMFRecommender:

    def test_recommend_returns_top_n(self, sample_mf_df):
        rec   = MutualFundRecommender()
        recs  = rec.recommend(sample_mf_df, risk_profile=RiskProfile.MODERATE, top_n=3)
        assert len(recs) <= 3

    def test_ranking_is_ordered(self, sample_mf_df):
        rec   = MutualFundRecommender()
        recs  = rec.recommend(sample_mf_df, risk_profile=RiskProfile.MODERATE)
        scores = [r.composite_score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_rank_sequential(self, sample_mf_df):
        rec  = MutualFundRecommender()
        recs = rec.recommend(sample_mf_df, risk_profile=RiskProfile.MODERATE)
        assert [r.rank for r in recs] == list(range(1, len(recs) + 1))

    def test_empty_df_returns_empty(self):
        rec  = MutualFundRecommender()
        recs = rec.recommend(pd.DataFrame(), risk_profile=RiskProfile.MODERATE)
        assert recs == []


# ─── FNO Analyzer Tests ──────────────────────────────────────────────────
class TestFNOAnalyzer:

    def test_pcr_calculation(self):
        analyzer = FNOAnalyzer()
        chain = {
            "calls": pd.DataFrame({"strike": [100, 110], "openInterest": [500, 300]}),
            "puts":  pd.DataFrame({"strike": [100, 110], "openInterest": [400, 500]}),
        }
        pcr = analyzer._compute_pcr(chain)
        assert 0 < pcr < 5

    def test_bullish_strategy(self):
        analyzer = FNOAnalyzer()
        pred = {"signal": "BUY", "confidence": 0.80, "current_price": 2000}
        rec  = analyzer.analyze("RELIANCE", pred, {}, {})
        assert "BUY" in rec.strategy.upper() or "BULL" in rec.strategy.upper()

    def test_bearish_strategy(self):
        analyzer = FNOAnalyzer()
        pred = {"signal": "SELL", "confidence": 0.75, "current_price": 2000}
        rec  = analyzer.analyze("RELIANCE", pred, {}, {})
        assert "PUT" in rec.strategy.upper() or "SELL" in rec.strategy.upper() or "BEAR" in rec.strategy.upper()


class TestTradingAgents:

    def test_sell_agent_detects_downside_for_portfolio(self, sample_predictions):
        agents = TradingAgents()
        plan = agents.create_trade_plan(
            predictions=sample_predictions,
            portfolio_symbols=["INFY.NS", "RELIANCE.NS"],
            top_n_buys=2,
        )
        sell_symbols = [a.symbol for a in plan["sell_actions"]]
        assert "INFY.NS" in sell_symbols

    def test_buy_agent_selects_growth_candidates(self, sample_predictions):
        agents = TradingAgents()
        plan = agents.create_trade_plan(
            predictions=sample_predictions,
            portfolio_symbols=["INFY.NS"],
            top_n_buys=2,
        )
        assert len(plan["buy_actions"]) > 0
        assert all(a.action == "BUY" for a in plan["buy_actions"])
        assert all(a.expected_return_5d >= 2.0 for a in plan["buy_actions"])

    def test_buy_agent_excludes_existing_portfolio(self, sample_predictions):
        agents = TradingAgents()
        plan = agents.create_trade_plan(
            predictions=sample_predictions,
            portfolio_symbols=["TCS.NS"],
            top_n_buys=5,
        )
        buy_symbols = [a.symbol for a in plan["buy_actions"]]
        assert "TCS.NS" not in buy_symbols
