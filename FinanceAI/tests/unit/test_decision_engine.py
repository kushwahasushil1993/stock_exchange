"""
src/market/decision_engine.py — the transparent, rule-based decision layer.
Covers the sign-consistency check added after the portfolio backtest
(src/pipelines/portfolio_backtest.py) surfaced real trades where Model A's
direction and Model B's expected return disagreed, and the decision engine
still called it a confident directional setup.
"""
from src.database.models import DecisionType, LiquidityLevel, MarketRegime, RiskLevel
from src.market.decision_engine import decide


def _base_kwargs(**overrides):
    kwargs = dict(
        probability_up=0.55, probability_down=0.40, probability_flat=0.05,
        expected_return=0.01, risk_level=RiskLevel.MEDIUM_RISK, liquidity_level=LiquidityLevel.HIGH,
        market_regime=MarketRegime.RANGE_BOUND,
    )
    kwargs.update(overrides)
    return kwargs


class TestSignConsistency:
    def test_model_a_up_model_b_positive_is_positive_setup(self):
        decision = decide(**_base_kwargs(probability_up=0.55, probability_down=0.40, expected_return=0.01))
        assert decision.decision == DecisionType.POSITIVE_SETUP

    def test_model_a_up_model_b_negative_is_no_clear_signal(self):
        """The exact disagreement pattern the portfolio backtest surfaced: probability favors UP but the return model predicts a loss."""
        decision = decide(**_base_kwargs(probability_up=0.55, probability_down=0.40, expected_return=-0.01))
        assert decision.decision == DecisionType.NO_CLEAR_SIGNAL
        assert any("disagree" in f for f in decision.negative_factors)

    def test_model_a_down_model_b_negative_is_negative_setup(self):
        decision = decide(**_base_kwargs(probability_up=0.35, probability_down=0.50, expected_return=-0.01))
        assert decision.decision == DecisionType.NEGATIVE_SETUP

    def test_model_a_down_model_b_positive_is_no_clear_signal(self):
        decision = decide(**_base_kwargs(probability_up=0.35, probability_down=0.50, expected_return=0.01))
        assert decision.decision == DecisionType.NO_CLEAR_SIGNAL
        assert any("disagree" in f for f in decision.negative_factors)


class TestExistingGates:
    def test_unsuitable_liquidity_is_no_clear_signal(self):
        decision = decide(**_base_kwargs(liquidity_level=LiquidityLevel.UNSUITABLE))
        assert decision.decision == DecisionType.NO_CLEAR_SIGNAL

    def test_weak_probability_edge_is_no_clear_signal(self):
        decision = decide(**_base_kwargs(probability_up=0.51, probability_down=0.49))
        assert decision.decision == DecisionType.NO_CLEAR_SIGNAL

    def test_tiny_expected_return_is_no_clear_signal(self):
        decision = decide(**_base_kwargs(expected_return=0.0001))
        assert decision.decision == DecisionType.NO_CLEAR_SIGNAL
