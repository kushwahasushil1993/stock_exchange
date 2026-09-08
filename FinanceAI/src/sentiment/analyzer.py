"""
Sentiment Analysis Pipeline:
  - FinBERT for financial text classification (Positive / Negative / Neutral)
  - Symbol extraction via NER + regex
  - Aggregated sentiment score per symbol
"""
import re
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)

# NSE top symbols (without .NS suffix) for entity matching
NSE_SYMBOLS = [
    "RELIANCE", "TCS", "INFY", "INFOSYS", "HDFCBANK", "ICICIBANK",
    "HINDUNILVR", "HUL", "SBIN", "STATEBANK", "BHARTIARTL", "AIRTEL",
    "ITC", "KOTAKBANK", "LT", "LARSEN", "HCLTECH", "WIPRO", "AXISBANK",
    "MARUTI", "BAJFINANCE", "TITAN", "NESTLEIND", "NESTLE", "ULTRACEMCO",
    "ASIANPAINT", "ADANIENT", "ADANIPORTS", "ADANIGREEN", "TATAMOTORS",
    "TATA", "TATASTEEL", "SUNPHARMA", "DRREDDY", "CIPLA", "POWERGRID",
    "NTPC", "ONGC", "COALINDIA", "TECHM", "EICHERMOT", "BAJAJFINSV",
    "GRASIM", "HINDALCO", "JSWSTEEL", "M&M", "MAHINDRA", "DIVISLAB",
    "APOLLOHOSP", "BPCL", "BRITANNIA", "INDUSINDBK", "UPL", "HERO",
    "HEROMOTOCO", "SHREECEM", "SBILIFE", "HDFCLIFE", "BAJAJ-AUTO",
    "NIFTY", "SENSEX", "BANKNIFTY", "NSE", "BSE",
]


class SymbolExtractor:
    """Extract stock symbols mentioned in news text."""

    def __init__(self):
        self._symbol_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(s) for s in NSE_SYMBOLS) + r')\b',
            re.IGNORECASE,
        )

    def extract(self, text: str) -> List[str]:
        found = set(m.upper() for m in self._symbol_pattern.findall(text or ""))
        return list(found)


class SentimentAnalyzer:
    """
    FinBERT-based sentiment analysis for financial text.
    Falls back to VADER (rule-based) if transformers not available.
    """

    def __init__(self, use_finbert: bool = True):
        self.use_finbert = use_finbert
        self._pipeline = None
        self._vader = None
        self._load_model()

    def _load_model(self):
        if self.use_finbert:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    device=-1,          # CPU; set to 0 for GPU
                    top_k=None,
                )
                logger.info("FinBERT loaded successfully")
                return
            except Exception as exc:
                logger.warning("FinBERT load failed (%s), falling back to VADER", exc)

        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer loaded")
        except ImportError:
            logger.error("Neither FinBERT nor VADER available. Install transformers or vaderSentiment.")

    def analyze(self, text: str) -> Dict:
        """
        Returns:
          {
            "label":  "positive" | "negative" | "neutral",
            "score":  float in [-1, 1]
          }
        """
        text = (text or "").strip()
        if not text:
            return {"label": "neutral", "score": 0.0}

        # Truncate to 512 tokens worth of characters
        text = text[:1024]

        if self._pipeline:
            return self._analyze_finbert(text)
        elif self._vader:
            return self._analyze_vader(text)
        return {"label": "neutral", "score": 0.0}

    def _analyze_finbert(self, text: str) -> Dict:
        try:
            results = self._pipeline(text)[0]
            label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
            # results is a list of {'label': ..., 'score': ...}
            best = max(results, key=lambda x: x["score"])
            label = best["label"].lower()
            magnitude = best["score"]
            scalar = label_map.get(label, 0.0) * magnitude
            return {"label": label, "score": round(scalar, 4)}
        except Exception as exc:
            logger.error("FinBERT inference error: %s", exc)
            return {"label": "neutral", "score": 0.0}

    def _analyze_vader(self, text: str) -> Dict:
        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return {"label": label, "score": round(compound, 4)}

    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        return [self.analyze(t) for t in texts]


class NewsAggregator:
    """
    Aggregate news sentiment per symbol with time-decay weighting.
    Recent news gets higher weight.
    """

    def __init__(self, decay_days: int = 3):
        self.extractor = SymbolExtractor()
        self.analyzer  = SentimentAnalyzer(use_finbert=True)
        self.decay_days = decay_days

    def process_articles(
        self,
        articles: List[Dict],
    ) -> List[Dict]:
        """
        Enrich each article with sentiment and extracted symbols.
        Returns the enriched list.
        """
        enriched = []
        for article in articles:
            text = f"{article.get('title', '')}. {article.get('content', '')}"
            sentiment = self.analyzer.analyze(text)
            symbols   = self.extractor.extract(text)

            enriched.append({
                **article,
                "sentiment_label": sentiment["label"],
                "sentiment_score": sentiment["score"],
                "symbols_mentioned": symbols,
            })
        return enriched

    def aggregate_by_symbol(
        self,
        enriched_articles: List[Dict],
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute a single sentiment score per symbol using time-decay weighted average.

        Returns: {symbol: score} where score ∈ [-1, 1]
        """
        now = datetime.utcnow()
        symbol_scores: Dict[str, List[Tuple[float, float]]] = {}  # symbol -> [(score, weight)]

        for article in enriched_articles:
            score   = article.get("sentiment_score", 0.0)
            s_list  = article.get("symbols_mentioned", [])

            # Time-based weight
            pub_str = article.get("published_at", "")
            try:
                pub_dt  = pd.to_datetime(pub_str, utc=True).to_pydatetime().replace(tzinfo=None)
                age_h   = max((now - pub_dt).total_seconds() / 3600, 1)
            except Exception:
                age_h = 24
            weight = np.exp(-age_h / (self.decay_days * 24))  # exponential decay

            for sym in s_list:
                clean = sym.upper().replace(".NS", "")
                if symbols and clean not in [s.upper().replace(".NS", "") for s in symbols]:
                    continue
                symbol_scores.setdefault(clean, []).append((score, weight))

        result = {}
        for sym, sw_list in symbol_scores.items():
            scores  = np.array([s for s, _ in sw_list])
            weights = np.array([w for _, w in sw_list])
            if weights.sum() > 0:
                result[sym] = float(np.average(scores, weights=weights))
            else:
                result[sym] = 0.0

        return result


# Lazy import fix for pd reference inside class
try:
    import pandas as pd
except ImportError:
    pass
