"""
Standalone script to run the full pipeline end-to-end without Docker:
  python scripts/run_pipeline.py
"""
import asyncio
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from src.ingestion.data_fetcher import YFinanceFetcher, NewsFetcher, AMFIFetcher
from src.features.technical import TechnicalFeatures, FeaturePipeline
from src.models.ensemble.predictor import ModelOrchestrator
from src.sentiment.analyzer import NewsAggregator
from src.recommendation.engine import StockScreener, FNOAnalyzer, MutualFundRecommender, RiskProfile
import pandas as pd


async def run():
    print("=" * 60)
    print("  FinanceAI — Full Pipeline Run")
    print("=" * 60)

    # ── 1. Fetch Market Data ─────────────────────────────────────────────
    print("\n📥 Step 1: Fetching market data…")
    yf      = YFinanceFetcher()
    symbols = settings.watchlist_nse[:5]   # Limit to 5 for demo

    ohlcv_data = {}
    fund_data  = {}
    for sym in symbols:
        print(f"  Fetching {sym}…", end=" ")
        ohlcv = yf.fetch_ohlcv(sym, period="2y")
        fund  = yf.fetch_fundamentals(sym)
        ohlcv_data[sym] = ohlcv
        fund_data[sym]  = fund
        print(f"✓ {len(ohlcv)} rows")

    # ── 2. News Sentiment ────────────────────────────────────────────────
    print("\n📰 Step 2: Fetching & analyzing news sentiment…")
    nf = NewsFetcher()
    try:
        articles = await nf.fetch_all(symbols)
        agg      = NewsAggregator()
        enriched = agg.process_articles(articles)
        sent_map = agg.aggregate_by_symbol(enriched, symbols)
        print(f"  Processed {len(enriched)} articles")
    except Exception as e:
        print(f"  ⚠️  News fetch failed: {e}")
        sent_map = {}

    # ── 3. Technical Features ────────────────────────────────────────────
    print("\n⚙️  Step 3: Computing technical indicators…")
    tf = TechnicalFeatures()
    for sym in symbols:
        if not ohlcv_data[sym].empty:
            enriched_df = tf.compute(ohlcv_data[sym])
            print(f"  {sym}: {len(enriched_df.columns)} features computed")

    # ── 4. Predictions ───────────────────────────────────────────────────
    print("\n🤖 Step 4: Running ensemble predictions…")
    orch      = ModelOrchestrator()
    all_preds = []

    for sym in symbols:
        if ohlcv_data[sym].empty:
            continue
        sym_clean = sym.replace(".NS", "")
        score     = sent_map.get(sym_clean, 0.0)
        pred      = orch.predict(sym, ohlcv_data[sym], fund_data[sym], score)
        all_preds.append(pred)
        print(f"  {sym:20s} | Signal: {pred['signal']:4s} | Confidence: {pred['confidence']:.1%} | "
              f"Target: ₹{pred['target_price']:.0f} | Stop: ₹{pred['stop_loss']:.0f}")

    # ── 5. Stock Recommendations ─────────────────────────────────────────
    print("\n🎯 Step 5: Top Stock Recommendations")
    screener = StockScreener()
    top_buys = screener.screen(all_preds, top_n=5, signal_filter="BUY")

    print(f"\n  {'Rank':<5} {'Symbol':<20} {'Score':<8} {'R:R':<6} {'Rationale'}")
    print("  " + "-" * 80)
    for r in top_buys:
        print(f"  #{r.rank:<4} {r.symbol:<20} {r.composite_score:.1%}   {r.risk_reward_ratio:<6.1f} {r.rationale[:50]}")

    # ── 6. FNO Analysis ──────────────────────────────────────────────────
    print("\n📊 Step 6: FNO Strategy Analysis")
    fno_analyzer = FNOAnalyzer()
    for pred in all_preds[:3]:
        sym  = pred["symbol"]
        rec  = fno_analyzer.analyze(sym, pred, {}, {})
        print(f"  {sym:20s} | Strategy: {rec.strategy:30s} | Risk: {rec.risk_level}")

    # ── 7. Mutual Fund Recommendations ───────────────────────────────────
    print("\n💰 Step 7: Mutual Fund Recommendations (MODERATE profile)")
    try:
        amfi       = AMFIFetcher()
        nav_df     = await amfi.fetch_all_navs()
        if not nav_df.empty:
            # Quick sample — fetch returns for top 50 schemes
            sample = nav_df.head(50).copy()
            sample["return_1y"]     = None
            sample["return_3y"]     = None
            sample["return_5y"]     = None
            sample["sharpe_ratio"]  = None
            sample["sortino_ratio"] = None
            sample["alpha"]         = None
            sample["expense_ratio"] = None
            sample["aum"]           = None

            mf_rec = MutualFundRecommender()
            recs   = mf_rec.recommend(sample, RiskProfile.MODERATE, top_n=3)
            for r in recs:
                print(f"  #{r.rank} {r.scheme_name[:50]:<50} | Score: {r.composite_score:.2%}")
        else:
            print("  ⚠️  Could not fetch AMFI data")
    except Exception as e:
        print(f"  ⚠️  MF recommendations failed: {e}")

    print("\n✅ Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run())
