"""
Recommendation Engine:
  1. Stock Screener & Ranker    — top picks by composite score
  2. FNO Analyzer               — futures/options strategy recommender
  3. Mutual Fund Recommender    — risk-adjusted MF ranking
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RiskProfile(str, Enum):
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE     = "MODERATE"
    AGGRESSIVE   = "AGGRESSIVE"


# ════════════════════════════════════════════════════════════════════════════
# Stock Screener & Ranker
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class StockRecommendation:
    symbol:              str
    signal:              str
    confidence:          float
    composite_score:     float
    technical_score:     float
    fundamental_score:   float
    sentiment_score:     float
    current_price:       float
    target_price:        float
    stop_loss:           float
    risk_reward_ratio:   float
    predicted_return_5d: float
    rationale:           str
    feature_importance:  Dict = field(default_factory=dict)
    rank:                int  = 0


class StockScreener:
    """
    Screen and rank stocks by composite score.
    Applies filters for liquidity, trend strength, and signal quality.
    """

    MIN_CONFIDENCE  = 0.60
    MIN_RR_RATIO    = 1.5
    MIN_ADX         = 20.0

    def screen(
        self,
        predictions: List[Dict],
        top_n: int = 10,
        signal_filter: Optional[str] = None,  # "BUY" | "SELL" | None
    ) -> List[StockRecommendation]:
        """
        Filter, rank, and return top-N stock recommendations.
        """
        candidates = []
        for p in predictions:
            if p.get("confidence", 0) < self.MIN_CONFIDENCE:
                continue
            if p.get("risk_reward_ratio", 0) < self.MIN_RR_RATIO:
                continue
            if signal_filter and p.get("signal") != signal_filter:
                continue

            rationale = self._build_rationale(p)
            candidates.append(StockRecommendation(
                symbol             = p["symbol"],
                signal             = p["signal"],
                confidence         = p["confidence"],
                composite_score    = p["composite_score"],
                technical_score    = p["technical_score"],
                fundamental_score  = p["fundamental_score"],
                sentiment_score    = p["sentiment_score"],
                current_price      = p["current_price"],
                target_price       = p["target_price"],
                stop_loss          = p["stop_loss"],
                risk_reward_ratio  = p["risk_reward_ratio"],
                predicted_return_5d= p["predicted_return_5d"],
                rationale          = rationale,
                feature_importance = p.get("feature_importance", {}),
            ))

        # Rank by composite_score DESC
        candidates.sort(key=lambda x: x.composite_score, reverse=True)
        for i, c in enumerate(candidates[:top_n], start=1):
            c.rank = i

        return candidates[:top_n]

    def _build_rationale(self, p: Dict) -> str:
        parts = []
        if p.get("technical_score", 0) > 0.7:
            parts.append("Strong technical setup")
        if p.get("fundamental_score", 0) > 0.7:
            parts.append("Sound fundamentals")
        if p.get("sentiment_score", 0) > 0.3:
            parts.append("Positive news sentiment")
        if p.get("ensemble_prob", 0) > 0.70:
            parts.append(f"High model confidence ({p['ensemble_prob']:.0%})")
        rr = p.get("risk_reward_ratio", 0)
        if rr >= 2:
            parts.append(f"Favorable R:R of {rr:.1f}x")
        return ". ".join(parts) if parts else "Based on composite model signals"


# ════════════════════════════════════════════════════════════════════════════
# Trading Agents (Sell-on-Downtrend + Future-Growth Buyer)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class TradeAction:
    symbol: str
    action: str
    confidence: float
    expected_return_5d: float
    composite_score: float
    rationale: str
    rank: int = 0


class SellOnDropAgent:
    """
    Generate SELL actions for existing positions when downside risk is elevated.
    """

    MIN_SELL_CONFIDENCE = 0.60
    MAX_EXPECTED_RETURN_5D = -2.0

    def evaluate(
        self,
        predictions: List[Dict],
        portfolio_symbols: List[str],
    ) -> List[TradeAction]:
        portfolio: Set[str] = {s.strip().upper() for s in portfolio_symbols if s}
        decisions: List[TradeAction] = []

        for p in predictions:
            symbol = str(p.get("symbol", "")).upper()
            if not symbol or symbol not in portfolio:
                continue

            signal = p.get("signal")
            confidence = float(p.get("confidence", 0.0) or 0.0)
            expected_return = float(p.get("predicted_return_5d", 0.0) or 0.0)
            composite = float(p.get("composite_score", 0.0) or 0.0)

            should_sell = (
                (signal == "SELL" and confidence >= self.MIN_SELL_CONFIDENCE)
                or expected_return <= self.MAX_EXPECTED_RETURN_5D
            )
            if not should_sell:
                continue

            rationale = (
                f"Downside risk detected: signal={signal}, "
                f"5D return={expected_return:.2f}%"
            )
            decisions.append(TradeAction(
                symbol=symbol,
                action="SELL",
                confidence=confidence,
                expected_return_5d=expected_return,
                composite_score=composite,
                rationale=rationale,
            ))

        decisions.sort(
            key=lambda d: (d.confidence, abs(min(d.expected_return_5d, 0.0))),
            reverse=True,
        )
        for i, d in enumerate(decisions, start=1):
            d.rank = i
        return decisions


class GrowthStockBuyerAgent:
    """
    Generate BUY actions for high-quality growth candidates.
    """

    MIN_BUY_CONFIDENCE = 0.65
    MIN_EXPECTED_RETURN_5D = 2.0
    MIN_RR_RATIO = 1.5

    def evaluate(
        self,
        predictions: List[Dict],
        top_n: int = 5,
        exclude_symbols: Optional[Set[str]] = None,
    ) -> List[TradeAction]:
        excluded = {s.upper() for s in (exclude_symbols or set())}
        candidates: List[TradeAction] = []

        for p in predictions:
            symbol = str(p.get("symbol", "")).upper()
            if not symbol or symbol in excluded:
                continue

            signal = p.get("signal")
            confidence = float(p.get("confidence", 0.0) or 0.0)
            expected_return = float(p.get("predicted_return_5d", 0.0) or 0.0)
            rr_ratio = float(p.get("risk_reward_ratio", 0.0) or 0.0)
            composite = float(p.get("composite_score", 0.0) or 0.0)

            if signal != "BUY":
                continue
            if confidence < self.MIN_BUY_CONFIDENCE:
                continue
            if expected_return < self.MIN_EXPECTED_RETURN_5D:
                continue
            if rr_ratio < self.MIN_RR_RATIO:
                continue

            rationale = (
                f"Growth setup confirmed: confidence={confidence:.2f}, "
                f"5D return={expected_return:.2f}%, R:R={rr_ratio:.2f}"
            )
            candidates.append(TradeAction(
                symbol=symbol,
                action="BUY",
                confidence=confidence,
                expected_return_5d=expected_return,
                composite_score=composite,
                rationale=rationale,
            ))

        candidates.sort(
            key=lambda d: (d.composite_score, d.expected_return_5d, d.confidence),
            reverse=True,
        )
        selected = candidates[:top_n]
        for i, d in enumerate(selected, start=1):
            d.rank = i
        return selected


class TradingAgents:
    """
    Orchestrates:
      1) SellOnDropAgent for current holdings
      2) GrowthStockBuyerAgent for re-allocation candidates
    """

    def __init__(
        self,
        sell_agent: Optional[SellOnDropAgent] = None,
        buy_agent: Optional[GrowthStockBuyerAgent] = None,
    ):
        self.sell_agent = sell_agent or SellOnDropAgent()
        self.buy_agent = buy_agent or GrowthStockBuyerAgent()

    def create_trade_plan(
        self,
        predictions: List[Dict],
        portfolio_symbols: List[str],
        top_n_buys: int = 5,
    ) -> Dict[str, List[TradeAction]]:
        sell_actions = self.sell_agent.evaluate(predictions, portfolio_symbols)
        exclude = {s.strip().upper() for s in portfolio_symbols if s}
        buy_actions = self.buy_agent.evaluate(
            predictions=predictions,
            top_n=top_n_buys,
            exclude_symbols=exclude,
        )
        return {
            "sell_actions": sell_actions,
            "buy_actions": buy_actions,
        }


# ════════════════════════════════════════════════════════════════════════════
# FNO Analyzer
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class FNORecommendation:
    symbol:          str
    strategy:        str       # "BUY FUTURE" | "SELL FUTURE" | "CALL" | "PUT" | "STRADDLE" | "BULL SPREAD"
    asset_type:      str       # "FUTURE" | "OPTION"
    expiry:          Optional[str]
    strike_price:    Optional[float]
    option_type:     Optional[str]    # CE | PE
    signal:          str
    confidence:      float
    iv:              Optional[float]
    pcr:             float
    oi_change_pct:   float
    max_pain:        Optional[float]
    rationale:       str
    risk_level:      str       # LOW | MEDIUM | HIGH


class FNOAnalyzer:
    """
    Analyze F&O data to generate derivative trading strategies:
    - Futures momentum
    - Options strategy based on IV, PCR, Max Pain
    - Open Interest buildup signals
    """

    PCR_BULLISH_THRESHOLD  = 0.7   # PCR < 0.7 → bullish
    PCR_BEARISH_THRESHOLD  = 1.3   # PCR > 1.3 → bearish
    IV_HIGH_THRESHOLD      = 30.0  # % - expensive options → spread strategies
    IV_LOW_THRESHOLD       = 15.0  # % - cheap options → directional buys

    def analyze(
        self,
        symbol: str,
        stock_prediction: Dict,
        option_chain: Dict,
        oi_data: Dict,
    ) -> FNORecommendation:
        """
        Generate FNO strategy recommendation.
        """
        signal      = stock_prediction.get("signal", "HOLD")
        confidence  = stock_prediction.get("confidence", 0.5)
        cur_price   = stock_prediction.get("current_price", 0)

        pcr          = self._compute_pcr(option_chain)
        max_pain     = self._compute_max_pain(option_chain, cur_price)
        iv           = self._get_atm_iv(option_chain, cur_price)
        oi_change    = self._get_oi_change(oi_data)

        strategy, rationale, risk = self._select_strategy(
            signal, confidence, pcr, iv, oi_change, cur_price, max_pain
        )

        # Nearest ATM strike
        atm_strike = round(cur_price / 50) * 50 if cur_price else None
        option_type = "CE" if signal == "BUY" else "PE"
        expiry = oi_data.get("expiry", option_chain.get("expiry"))

        return FNORecommendation(
            symbol       = symbol,
            strategy     = strategy,
            asset_type   = "OPTION" if "CE" in strategy or "PE" in strategy or "STRADDLE" in strategy else "FUTURE",
            expiry       = str(expiry) if expiry else None,
            strike_price = atm_strike,
            option_type  = option_type if "OPTION" in (strategy.upper()) else None,
            signal       = signal,
            confidence   = confidence,
            iv           = iv,
            pcr          = pcr,
            oi_change_pct= oi_change,
            max_pain     = max_pain,
            rationale    = rationale,
            risk_level   = risk,
        )

    def _compute_pcr(self, chain: Dict) -> float:
        """Put-Call Ratio by Open Interest."""
        try:
            calls = pd.DataFrame(chain.get("calls", []))
            puts  = pd.DataFrame(chain.get("puts", []))
            put_oi  = puts["openInterest"].sum()  if "openInterest" in puts.columns  else 1
            call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 1
            return round(put_oi / max(call_oi, 1), 2)
        except Exception:
            return 1.0

    def _compute_max_pain(self, chain: Dict, spot: float) -> Optional[float]:
        """
        Max Pain = strike at which option sellers lose the least.
        At expiry, most options (by OI) expire worthless.
        """
        try:
            calls = pd.DataFrame(chain.get("calls", []))
            puts  = pd.DataFrame(chain.get("puts", []))
            if calls.empty or "strike" not in calls.columns:
                return None

            strikes = sorted(set(
                list(calls["strike"].unique()) + list(puts["strike"].unique())
            ))
            pain = {}
            for s in strikes:
                # Loss to call writers
                call_loss = calls[calls["strike"] < s].apply(
                    lambda r: max(s - r["strike"], 0) * r.get("openInterest", 0), axis=1
                ).sum()
                # Loss to put writers
                put_loss = puts[puts["strike"] > s].apply(
                    lambda r: max(r["strike"] - s, 0) * r.get("openInterest", 0), axis=1
                ).sum()
                pain[s] = call_loss + put_loss

            return min(pain, key=pain.get) if pain else None
        except Exception:
            return None

    def _get_atm_iv(self, chain: Dict, spot: float) -> Optional[float]:
        try:
            calls = pd.DataFrame(chain.get("calls", []))
            if calls.empty or "impliedVolatility" not in calls.columns:
                return None
            atm_idx = (calls["strike"] - spot).abs().idxmin()
            iv_val  = calls.loc[atm_idx, "impliedVolatility"] * 100
            return round(float(iv_val), 2)
        except Exception:
            return None

    def _get_oi_change(self, oi_data: Dict) -> float:
        try:
            return float(oi_data.get("oiChange", oi_data.get("oi_change_pct", 0.0)))
        except Exception:
            return 0.0

    def _select_strategy(
        self,
        signal: str, confidence: float,
        pcr: float, iv: Optional[float],
        oi_change: float, price: float,
        max_pain: Optional[float],
    ) -> Tuple[str, str, str]:
        iv_val = iv or 20.0

        if signal == "BUY" and confidence > 0.70:
            if iv_val < self.IV_LOW_THRESHOLD:
                return ("BUY CALL OPTION", "Low IV + bullish signal → buy ATM call", "MEDIUM")
            elif iv_val > self.IV_HIGH_THRESHOLD:
                return ("BULL CALL SPREAD", "High IV + bullish signal → limit premium with spread", "LOW")
            else:
                return ("BUY FUTURE", "Strong bullish signal with normal IV → futures momentum trade", "HIGH")

        elif signal == "SELL" and confidence > 0.70:
            if iv_val < self.IV_LOW_THRESHOLD:
                return ("BUY PUT OPTION", "Low IV + bearish signal → buy ATM put", "MEDIUM")
            elif iv_val > self.IV_HIGH_THRESHOLD:
                return ("BEAR PUT SPREAD", "High IV + bearish signal → limit premium with spread", "LOW")
            else:
                return ("SELL FUTURE", "Strong bearish signal → futures short", "HIGH")

        elif iv_val > 35 and 0.9 < pcr < 1.1:
            return ("IRON CONDOR", "High IV + neutral PCR → premium collection strategy", "LOW")

        elif max_pain and abs(price - max_pain) / max_pain < 0.02:
            return ("SHORT STRADDLE", "Price near max pain → time decay strategy", "MEDIUM")

        else:
            return ("HOLD / WAIT", "No clear FNO signal — wait for confirmation", "LOW")


# ════════════════════════════════════════════════════════════════════════════
# Mutual Fund Recommender
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class MFRecommendation:
    scheme_code:      str
    scheme_name:      str
    category:         str
    amc:              str
    nav:              float
    return_1y:        Optional[float]
    return_3y:        Optional[float]
    return_5y:        Optional[float]
    sharpe_ratio:     Optional[float]
    sortino_ratio:    Optional[float]
    alpha:            Optional[float]
    expense_ratio:    Optional[float]
    composite_score:  float
    rank:             int
    risk_profile:     str
    rationale:        str
    aum:              Optional[float]


class MutualFundRecommender:
    """
    Score and rank mutual funds by:
      - Risk-adjusted returns (Sharpe, Sortino)
      - Alpha over benchmark
      - Consistency of returns
      - Expense ratio (lower is better)
      - AUM (stability indicator)
    """

    WEIGHTS = {
        "return_3y":      0.25,
        "return_5y":      0.20,
        "sharpe_ratio":   0.20,
        "sortino_ratio":  0.15,
        "alpha":          0.15,
        "expense_ratio":  0.05,  # inverted
    }

    def recommend(
        self,
        mf_df: pd.DataFrame,
        risk_profile: RiskProfile = RiskProfile.MODERATE,
        top_n: int = 5,
        category_filter: Optional[str] = None,
    ) -> List[MFRecommendation]:
        """
        mf_df must have columns: scheme_code, scheme_name, category, amc, nav,
          return_1y, return_3y, return_5y, sharpe_ratio, sortino_ratio,
          alpha, expense_ratio, aum.
        """
        df = mf_df.copy()

        # Filter by risk profile
        # AMFI's raw category strings are verbose, e.g.
        # "Open Ended Schemes(Equity Scheme - Large Cap Fund)" — never equal
        # to the short labels below, so match by substring instead of isin().
        profile_categories = self._get_profile_categories(risk_profile)
        df = df[df["category"].apply(
            lambda c: any(pc.lower() in str(c).lower() for pc in profile_categories)
        )]

        if category_filter:
            df = df[df["category"].str.contains(category_filter, case=False, na=False)]

        if df.empty:
            return []

        # Compute composite score
        df["composite_score"] = df.apply(
            lambda row: self._score_fund(row), axis=1
        )

        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

        recs = []
        for i, row in df.head(top_n).iterrows():
            recs.append(MFRecommendation(
                scheme_code     = str(row.get("scheme_code", "")),
                scheme_name     = str(row.get("scheme_name", "")),
                category        = str(row.get("category", "")),
                amc             = str(row.get("amc", "")),
                nav             = float(row.get("nav") or 0),
                return_1y       = self._safe_float(row.get("return_1y")),
                return_3y       = self._safe_float(row.get("return_3y")),
                return_5y       = self._safe_float(row.get("return_5y")),
                sharpe_ratio    = self._safe_float(row.get("sharpe_ratio")),
                sortino_ratio   = self._safe_float(row.get("sortino_ratio")),
                alpha           = self._safe_float(row.get("alpha")),
                expense_ratio   = self._safe_float(row.get("expense_ratio")),
                composite_score = round(float(row["composite_score"]), 4),
                rank            = i + 1,
                risk_profile    = risk_profile.value,
                rationale       = self._build_rationale(row),
                aum             = self._safe_float(row.get("aum")),
            ))

        return recs

    def _score_fund(self, row: pd.Series) -> float:
        score = 0.0
        # Normalize each metric to [0, 1] using reasonable market ranges
        metrics = {
            "return_3y":    (row.get("return_3y")    or 0, -5,  30),
            "return_5y":    (row.get("return_5y")    or 0, -5,  25),
            "sharpe_ratio": (row.get("sharpe_ratio") or 0,  0,   3),
            "sortino_ratio":(row.get("sortino_ratio")or 0,  0,   4),
            "alpha":        (row.get("alpha")         or 0, -5,  15),
        }
        for key, (val, lo, hi) in metrics.items():
            if val is not None:
                normalized = np.clip((float(val) - lo) / (hi - lo), 0, 1)
                score += self.WEIGHTS.get(key, 0) * normalized

        # Expense ratio: lower is better (invert)
        er = row.get("expense_ratio") or 2.5
        er_score = np.clip(1 - (float(er) / 2.5), 0, 1)
        score += self.WEIGHTS["expense_ratio"] * er_score

        return float(score)

    def _get_profile_categories(self, profile: RiskProfile) -> List[str]:
        mapping = {
            RiskProfile.CONSERVATIVE: [
                "Debt - Short Duration", "Debt - Banking and PSU",
                "Liquid", "Overnight", "Arbitrage",
                "Debt - Low Duration", "Debt - Money Market",
            ],
            RiskProfile.MODERATE: [
                "Large Cap", "Balanced Advantage", "Aggressive Hybrid",
                "Index Fund", "Flexi Cap", "Multi Cap",
            ],
            RiskProfile.AGGRESSIVE: [
                "Mid Cap", "Small Cap", "Sectoral/Thematic",
                "ELSS", "Focused Fund", "Value Fund",
            ],
        }
        return mapping.get(profile, mapping[RiskProfile.MODERATE])

    def _build_rationale(self, row: pd.Series) -> str:
        parts = []
        r3 = row.get("return_3y")
        r5 = row.get("return_5y")
        if r3 and r3 > 15:
            parts.append(f"Strong 3Y returns of {r3:.1f}%")
        if r5 and r5 > 12:
            parts.append(f"Consistent 5Y CAGR of {r5:.1f}%")
        sr = row.get("sharpe_ratio")
        if sr and sr > 1.5:
            parts.append(f"High Sharpe ratio of {sr:.2f}")
        alpha = row.get("alpha")
        if alpha and alpha > 3:
            parts.append(f"Outperforms benchmark by {alpha:.1f}%")
        er = row.get("expense_ratio")
        if er and er < 1.0:
            parts.append(f"Low expense ratio of {er:.2f}%")
        return ". ".join(parts) if parts else "Well-diversified fund with stable returns"

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        try:
            return float(val) if val is not None else None
        except (ValueError, TypeError):
            return None
