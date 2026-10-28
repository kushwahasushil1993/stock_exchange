"""
Real NSE sectoral indices — the `sector_return` / `sector_relative_strength`
feature pair, mapped from each Stock's own yfinance-reported `sector` (GICS-
style taxonomy: Technology, Financial Services, Healthcare, ...) to the
closest REAL NSE sectoral index tracked on Yahoo Finance.

IMPORTANT, VERIFIED-NOT-ASSUMED LIMITATION: yfinance resolves tickers for
several more NSE sectoral indices than are usable here (^CNXAUTO, ^CNXFMCG,
^CNXMETAL, ^CNXENERGY, ^CNXREALTY, ^CNXMEDIA, ^CNXINFRA all return a valid
quote) — but carry NO historical daily time series at all: even
`history(period="max")` returns exactly ONE row, today's snapshot, for
every one of them. That's not a rate limit or a transient gap; it's Yahoo
simply not carrying history for these specific index tickers. A feature
computed from a single data point isn't a feature — those sectors are left
UNMAPPED rather than quietly building on fabricated-by-omission history.
Likewise NIFTY_FIN_SERVICE.NS (already scanned elsewhere as FINNIFTY) has
only 1 real historical row, so "Financial Services" uses NIFTY BANK
instead — narrower than the full GICS category, but has genuine multi-year
history (1235 real daily bars, verified).

Only THREE sectors currently have a real, usable (1200+ real daily bars,
verified live) NSE sectoral index behind them:
  - Technology            -> NIFTY IT      (^CNXIT)
  - Healthcare             -> NIFTY Pharma  (^CNXPHARMA)
  - Financial Services     -> NIFTY Bank    (^NSEBANK) — an approximation:
    GICS "Financial Services" also covers NBFCs/insurance that NIFTY BANK
    doesn't track, but it's the only real, historically-usable proxy
    available for this sector.

The other 7 sectors present in this DB (Consumer Cyclical, Basic
Materials, Industrials, Consumer Defensive, Real Estate, Communication
Services, Energy) and Utilities get NO sector_return/sector_relative_
strength feature (NaN) rather than a fabricated one, until either
yfinance's coverage improves or a different data provider carries this
history — a real, documented gap, not a silently degraded one.
"""
from typing import Optional

SECTOR_INDEX_MAP = {
    "Technology":         {"yf_symbol": "^CNXIT",     "index_name": "NIFTY IT",     "approximate": False},
    "Healthcare":         {"yf_symbol": "^CNXPHARMA", "index_name": "NIFTY Pharma", "approximate": False},
    "Financial Services": {"yf_symbol": "^NSEBANK",   "index_name": "NIFTY Bank",   "approximate": True},
    # Every other sector present in this DB (Consumer Cyclical, Basic
    # Materials, Industrials, Consumer Defensive, Real Estate,
    # Communication Services, Energy, Utilities) is intentionally absent —
    # see module docstring for the verified reason (no real yfinance
    # history, or no index at all).
}

# ^NSEBANK is already scanned every run as BANKNIFTY (see
# src.ingestion.data_fetcher.INDEX_FNO) — no need to fetch/persist it twice.
_ALREADY_SCANNED_ELSEWHERE = {"^NSEBANK"}

SECTOR_INDEX_SYMBOLS = sorted({
    cfg["yf_symbol"] for cfg in SECTOR_INDEX_MAP.values()
    if cfg["yf_symbol"] not in _ALREADY_SCANNED_ELSEWHERE
})


def get_sector_index_symbol(sector: Optional[str]) -> Optional[str]:
    """The yfinance ticker for `sector`'s real NSE sectoral-index proxy, or None if `sector` is missing or has no defensible mapping (e.g. Utilities)."""
    if not sector:
        return None
    cfg = SECTOR_INDEX_MAP.get(sector)
    return cfg["yf_symbol"] if cfg else None
