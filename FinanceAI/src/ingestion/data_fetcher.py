"""
Multi-source data ingestion:
  - Yahoo Finance (yfinance) – OHLCV, fundamentals
  - NSE India REST API     – live quotes, F&O OI, PCR
  - NewsAPI                – financial news
  - Moneycontrol RSS       – Indian market news
  - AMFI API               – Mutual fund NAV & returns
"""
import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
import json

from config.settings import settings

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Yahoo Finance Fetcher
# ════════════════════════════════════════════════════════════════════════════
class YFinanceFetcher:
    """Fetch OHLCV data and fundamentals from Yahoo Finance."""

    def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
        Open, High, Low, Close, Volume, Adj Close
        indexed by Date.
        """
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            logger.warning("No data returned for %s", symbol)
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df["symbol"] = symbol
        return df

    def fetch_multiple(
        self,
        symbols: List[str],
        period: str = "2y",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.fetch_ohlcv(sym, period, interval)
            except Exception as exc:
                logger.error("YFinance error for %s: %s", sym, exc)
        return result

    def fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        return {
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "revenue_growth_yoy": info.get("revenueGrowth"),
            "profit_margin": info.get("profitMargins"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "free_cash_flow": info.get("freeCashflow"),
            "eps_growth": info.get("earningsGrowth"),
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "company_name": info.get("longName"),
        }

    def fetch_options_chain(self, symbol: str) -> Dict[str, pd.DataFrame]:
        ticker = yf.Ticker(symbol)
        expiries = ticker.options
        if not expiries:
            return {}
        # Fetch nearest expiry
        nearest = expiries[0]
        chain = ticker.option_chain(nearest)
        return {
            "calls": chain.calls,
            "puts": chain.puts,
            "expiry": nearest,
        }


# ════════════════════════════════════════════════════════════════════════════
# NSE India Fetcher (async)
# ════════════════════════════════════════════════════════════════════════════
class NSEFetcher:
    """
    Fetch live data from NSE India public APIs.
    Headers mimic browser to avoid 403.
    """
    BASE_URL = "https://www.nseindia.com"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/",
    }

    async def _get_session_cookie(self, session: aiohttp.ClientSession) -> str:
        """Obtain NSE session cookie required for API calls."""
        async with session.get(self.BASE_URL, headers=self.HEADERS) as resp:
            cookies = {k: v.value for k, v in resp.cookies.items()}
            return "; ".join([f"{k}={v}" for k, v in cookies.items()])

    async def fetch_quote(self, symbol: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/api/quote-equity?symbol={symbol}"
        async with aiohttp.ClientSession() as session:
            cookie = await self._get_session_cookie(session)
            headers = {**self.HEADERS, "Cookie": cookie}
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {}

    async def fetch_fno_oi(self, symbol: str) -> Dict[str, Any]:
        """Fetch Open Interest data for F&O stocks."""
        url = f"{self.BASE_URL}/api/quote-derivative?symbol={symbol}"
        async with aiohttp.ClientSession() as session:
            cookie = await self._get_session_cookie(session)
            headers = {**self.HEADERS, "Cookie": cookie}
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {}

    async def fetch_option_chain(self, symbol: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/api/option-chain-equities?symbol={symbol}"
        async with aiohttp.ClientSession() as session:
            cookie = await self._get_session_cookie(session)
            headers = {**self.HEADERS, "Cookie": cookie}
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {}

    async def fetch_indices(self) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/api/allIndices"
        async with aiohttp.ClientSession() as session:
            cookie = await self._get_session_cookie(session)
            headers = {**self.HEADERS, "Cookie": cookie}
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {}


# ════════════════════════════════════════════════════════════════════════════
# News Fetcher
# ════════════════════════════════════════════════════════════════════════════
class NewsFetcher:
    """Aggregate financial news from multiple sources."""

    NEWSAPI_URL = "https://newsapi.org/v2/everything"
    MONEYCONTROL_RSS = "https://www.moneycontrol.com/rss/marketreports.xml"
    ECONOMIC_TIMES_RSS = "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"

    def __init__(self):
        self.api_key = settings.newsapi_key

    async def fetch_newsapi(
        self,
        query: str = "Indian stock market NSE BSE Nifty",
        days_back: int = 3,
        page_size: int = 100,
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            logger.warning("NewsAPI key not set, skipping NewsAPI fetch")
            return []

        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": page_size,
            "apiKey": self.api_key,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(self.NEWSAPI_URL, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("articles", [])
                logger.error("NewsAPI error: %s", resp.status)
                return []

    async def fetch_rss(self, url: str) -> List[Dict[str, Any]]:
        """Parse RSS feed and return list of article dicts."""
        import feedparser
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    content = await resp.text()
                    feed = feedparser.parse(content)
                    articles = []
                    for entry in feed.entries:
                        articles.append({
                            "title": entry.get("title", ""),
                            "url": entry.get("link", ""),
                            "content": entry.get("summary", ""),
                            "published_at": entry.get("published", ""),
                            "source": feed.feed.get("title", url),
                        })
                    return articles
        return []

    async def fetch_all(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Fetch from all news sources concurrently."""
        queries = ["Indian stock market", "NSE BSE Nifty Sensex"]
        if symbols:
            # Add top 5 symbols to query
            queries.append(" OR ".join(s.replace(".NS", "") for s in symbols[:5]))

        tasks = []
        for q in queries:
            tasks.append(self.fetch_newsapi(query=q))
        tasks.append(self.fetch_rss(self.MONEYCONTROL_RSS))
        tasks.append(self.fetch_rss(self.ECONOMIC_TIMES_RSS))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        articles = []
        for r in results:
            if isinstance(r, list):
                articles.extend(r)
            else:
                logger.warning("News fetch error: %s", r)

        # Deduplicate by URL
        seen = set()
        unique = []
        for a in articles:
            url = a.get("url", "")
            if url and url not in seen:
                seen.add(url)
                unique.append(a)

        return unique


# ════════════════════════════════════════════════════════════════════════════
# AMFI Mutual Fund Fetcher
# ════════════════════════════════════════════════════════════════════════════
class AMFIFetcher:
    """
    Fetch Mutual Fund NAV data from AMFI India (free, no API key needed).
    Returns structured data for scoring and recommendation.
    """
    NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
    MFAPI_BASE = "https://api.mfapi.in/mf"

    async def fetch_all_navs(self) -> pd.DataFrame:
        """Download all scheme NAVs from AMFI."""
        async with aiohttp.ClientSession() as session:
            async with session.get(self.NAV_URL, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    text = await resp.text(encoding="utf-8", errors="ignore")
                    return self._parse_amfi_nav(text)
        return pd.DataFrame()

    def _parse_amfi_nav(self, raw: str) -> pd.DataFrame:
        rows = []
        current_category = ""
        current_amc = ""
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("Open Ended") or line.startswith("Close Ended"):
                current_category = line
            elif ";" not in line:
                current_amc = line
            else:
                parts = line.split(";")
                if len(parts) >= 6:
                    rows.append({
                        "scheme_code": parts[0].strip(),
                        "isin_div_payout": parts[1].strip(),
                        "isin_div_reinvest": parts[2].strip(),
                        "scheme_name": parts[3].strip(),
                        "nav": self._safe_float(parts[4]),
                        "date": parts[5].strip(),
                        "category": current_category,
                        "amc": current_amc,
                    })
        return pd.DataFrame(rows)

    async def fetch_historical_nav(self, scheme_code: str) -> pd.DataFrame:
        """Fetch historical NAV for a scheme from mfapi.in."""
        url = f"{self.MFAPI_BASE}/{scheme_code}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    navs = data.get("data", [])
                    df = pd.DataFrame(navs)
                    if not df.empty:
                        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
                        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
                        df.sort_values("date", inplace=True)
                        df.reset_index(drop=True, inplace=True)
                    return df
        return pd.DataFrame()

    def compute_returns(self, df: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Compute 1Y, 3Y, 5Y CAGR from historical NAV DataFrame."""
        if df.empty or "nav" not in df.columns:
            return {}
        df = df.dropna(subset=["nav"]).sort_values("date")
        latest = df.iloc[-1]
        results = {}
        for label, years in [("return_1y", 1), ("return_3y", 3), ("return_5y", 5)]:
            target_date = latest["date"] - timedelta(days=int(years * 365.25))
            hist = df[df["date"] <= target_date]
            if not hist.empty:
                past = hist.iloc[-1]["nav"]
                if past and past > 0:
                    cagr = (latest["nav"] / past) ** (1 / years) - 1
                    results[label] = round(cagr * 100, 2)
                else:
                    results[label] = None
            else:
                results[label] = None
        return results

    @staticmethod
    def _safe_float(val: str) -> Optional[float]:
        try:
            return float(val.strip())
        except (ValueError, AttributeError):
            return None


# ════════════════════════════════════════════════════════════════════════════
# Macro Economic Data Fetcher
# ════════════════════════════════════════════════════════════════════════════
class MacroFetcher:
    """
    Fetch macro indicators: USD/INR, Gold, Crude Oil, US10Y yield.
    These are used as model features.
    """

    MACRO_SYMBOLS = {
        "usd_inr": "USDINR=X",
        "gold": "GC=F",
        "crude_oil": "CL=F",
        "us10y_yield": "^TNX",
        "vix_india": "^INDIAVIX",
        "sp500": "^GSPC",
        "dxy": "DX-Y.NYB",
    }

    def fetch_all(self, period: str = "1y") -> Dict[str, pd.DataFrame]:
        fetcher = YFinanceFetcher()
        return fetcher.fetch_multiple(
            list(self.MACRO_SYMBOLS.values()), period=period
        )
