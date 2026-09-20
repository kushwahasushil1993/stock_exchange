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
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
        except Exception as exc:
            # Yahoo rate-limits (HTTP 429) aggressively and yfinance sometimes
            # surfaces that as a raw JSONDecodeError instead of a clean error —
            # degrade gracefully instead of crashing the caller.
            logger.warning("Failed to fetch fundamentals for %s: %s", symbol, exc)
            info = {}

        def _finite(val: Any) -> Optional[float]:
            # Yahoo sometimes returns the literal string "Infinity"/"NaN", or a
            # real inf/nan float, for metrics like PE when earnings are ~0 —
            # none of those are valid Postgres float params.
            try:
                f = float(val)
            except (TypeError, ValueError):
                return None
            return f if np.isfinite(f) else None

        return {
            "pe_ratio": _finite(info.get("trailingPE")),
            "pb_ratio": _finite(info.get("priceToBook")),
            "roe": _finite(info.get("returnOnEquity")),
            "debt_to_equity": _finite(info.get("debtToEquity")),
            "revenue_growth_yoy": _finite(info.get("revenueGrowth")),
            "profit_margin": _finite(info.get("profitMargins")),
            "dividend_yield": _finite(info.get("dividendYield")),
            "beta": _finite(info.get("beta")),
            "free_cash_flow": _finite(info.get("freeCashflow")),
            "eps_growth": _finite(info.get("earningsGrowth")),
            "market_cap": _finite(info.get("marketCap")),
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
# NSE Universe Fetcher — full listed-equity / F&O-eligible symbol lists
# ════════════════════════════════════════════════════════════════════════════
class NSEUniverseFetcher:
    """
    Fetch the full universe of NSE-listed symbols from NSE's public archive
    (static files, no session/cookie dance needed — unlike the live NSE API).
    """
    EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    FNO_LIST_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    # Index derivatives that appear in fo_mktlots.csv alongside single-stock ones
    INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "NIFTYFPI"}

    async def fetch_equity_symbols(self) -> List[Dict[str, str]]:
        """
        All NSE main-board (SERIES == EQ) listed equities.
        Returns [{"symbol": "RELIANCE", "yf_symbol": "RELIANCE.NS", "company_name": "..."}]
        """
        import csv
        import io

        async with aiohttp.ClientSession() as session:
            async with session.get(self.EQUITY_LIST_URL, headers=self.HEADERS) as resp:
                text = await resp.text()

        reader = csv.DictReader(io.StringIO(text))
        result = []
        for row in reader:
            row = {k.strip(): (v.strip() if v else v) for k, v in row.items()}
            if row.get("SERIES") != "EQ":
                continue
            symbol = row.get("SYMBOL", "")
            if not symbol:
                continue
            result.append({
                "symbol": symbol,
                "yf_symbol": f"{symbol}.NS",
                "company_name": row.get("NAME OF COMPANY", symbol),
            })
        return result

    async def fetch_fno_symbols(self) -> List[str]:
        """
        All single-stock F&O-eligible symbols (index futures like NIFTY/BANKNIFTY
        excluded — those aren't equity tickers). Returns yfinance-formatted symbols.
        """
        import csv
        import io

        async with aiohttp.ClientSession() as session:
            async with session.get(self.FNO_LIST_URL, headers=self.HEADERS) as resp:
                text = await resp.text()

        reader = csv.reader(io.StringIO(text))
        next(reader, None)  # header row
        symbols = []
        for row in reader:
            if len(row) < 2:
                continue
            sym = row[1].strip()
            if not sym or sym.upper() in {"SYMBOL"} or sym.upper() in self.INDEX_SYMBOLS:
                continue
            symbols.append(f"{sym}.NS")
        return symbols


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
        """
        Obtain the NSE session cookie required for API calls.

        Uses NSE_COOKIE from settings if configured (a cookie captured from a
        real browser session on nseindia.com — Akamai's bot manager otherwise
        blocks plain server-to-server requests). It expires within hours, so
        this is a stopgap; falls back to a fresh dynamic fetch when unset or
        once NSE stops honoring it.
        """
        if settings.nse_cookie:
            return settings.nse_cookie
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
        """
        AMFI's column layout has changed over time (older files had 6
        semicolon fields: code;isin;isin;name;nav;date — the current live
        feed has 8, with `Plan`/`Option` inserted before nav/date). Parse via
        the header row's column names rather than hardcoded positions so a
        future column reshuffle doesn't silently turn every `nav` into None.
        """
        rows = []
        current_category = ""
        current_amc = ""
        header: Optional[List[str]] = None
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("Open Ended") or line.startswith("Close Ended"):
                current_category = line
                continue
            if ";" not in line:
                current_amc = line
                continue

            parts = [p.strip() for p in line.split(";")]
            if parts[0] == "Scheme Code":
                header = parts  # header row — capture column order, don't emit a row
                continue

            if header and len(parts) == len(header):
                row = dict(zip(header, parts))
            elif len(parts) >= 6:
                # Fallback for the legacy 6-column format
                row = {
                    "Scheme Code": parts[0],
                    "ISIN Div Payout/ ISIN Growth": parts[1],
                    "ISIN Div Reinvestment": parts[2],
                    "Scheme Name": parts[3],
                    "Net Asset Value": parts[4],
                    "Date": parts[5],
                }
            else:
                continue

            rows.append({
                "scheme_code": row.get("Scheme Code", ""),
                "isin_div_payout": row.get("ISIN Div Payout/ ISIN Growth", ""),
                "isin_div_reinvest": row.get("ISIN Div Reinvestment", ""),
                "scheme_name": row.get("Scheme Name", ""),
                "plan": row.get("Plan", ""),
                "option": row.get("Option", ""),
                "nav": self._safe_float(row.get("Net Asset Value", "")),
                "date": row.get("Date", ""),
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
