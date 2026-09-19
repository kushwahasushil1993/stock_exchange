"""Mutual Funds router — top MF recommendations by risk profile."""
import asyncio
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import List, Optional
from src.api.routers.auth import get_current_user, User
from src.recommendation.engine import MFRecommendation, MutualFundRecommender, RiskProfile

router = APIRouter()


async def get_top_mutual_funds(
    risk_profile: str = "MODERATE",
    top_n: int = 5,
    category: Optional[str] = None,
) -> List[MFRecommendation]:
    """
    Shared MF ranking logic — used by both /recommendations and the
    consolidated /top-buys endpoint. This is a live snapshot (AMFI has no
    dated "prediction" history to filter by, unlike stocks/F&O), so it takes
    no date range.
    """
    from src.ingestion.data_fetcher import AMFIFetcher
    import pandas as pd

    fetcher = AMFIFetcher()
    nav_df = await fetcher.fetch_all_navs()
    if nav_df.empty:
        return []

    # Filter to open-ended schemes in categories relevant to the requested
    # risk profile *before* sampling. AMFI's file is ordered alphabetically by
    # category, so a plain .head(200) over the unfiltered universe can easily
    # sample 200 schemes with zero matches for whatever profile was asked for
    # (e.g. MODERATE wants Equity/Hybrid categories, which sort well after
    # "Children's Fund"/"Debt Scheme..." alphabetically).
    profile_categories = MutualFundRecommender()._get_profile_categories(RiskProfile(risk_profile))
    relevant = nav_df[
        nav_df["category"].str.contains("Open Ended", na=False)
        & nav_df["category"].apply(lambda c: any(pc.lower() in str(c).lower() for pc in profile_categories))
    ]
    if category:
        relevant = relevant[relevant["category"].str.contains(category, case=False, na=False)]

    # Enrich with computed returns (sample up to 200 matching funds).
    # Bounded concurrency: 200 sequential requests to mfapi.in would take
    # minutes; a plain asyncio.gather with no cap would fire all 200 at once.
    open_ended = relevant.head(200)
    sem = asyncio.Semaphore(20)

    async def _fetch_with_returns(row):
        async with sem:
            hist = await fetcher.fetch_historical_nav(str(row["scheme_code"]))
        return {**row.to_dict(), **fetcher.compute_returns(hist)}

    enriched_rows = await asyncio.gather(*(
        _fetch_with_returns(row) for _, row in open_ended.iterrows()
    ))

    enriched_df = pd.DataFrame(enriched_rows)

    # Fill missing risk metrics with reasonable defaults
    enriched_df["sharpe_ratio"]  = None
    enriched_df["sortino_ratio"] = None
    enriched_df["alpha"]         = None
    enriched_df["expense_ratio"] = None
    enriched_df["aum"]           = None

    return MutualFundRecommender().recommend(
        enriched_df,
        risk_profile    = RiskProfile(risk_profile),
        top_n           = top_n,
        category_filter = category,
    )


class MFOut(BaseModel):
    rank:            int
    scheme_code:     str
    scheme_name:     str
    category:        str
    amc:             str
    nav:             float
    return_1y:       Optional[float]
    return_3y:       Optional[float]
    return_5y:       Optional[float]
    sharpe_ratio:    Optional[float]
    expense_ratio:   Optional[float]
    composite_score: float
    risk_profile:    str
    rationale:       str
    aum:             Optional[float]


@router.get("/recommendations", response_model=List[MFOut])
async def mf_recommendations(
    risk_profile: str = Query(
        default="MODERATE",
        pattern="^(CONSERVATIVE|MODERATE|AGGRESSIVE)$",
    ),
    category:  Optional[str] = Query(default=None),
    top_n:     int = Query(default=5, ge=1, le=20),
    _user:     User = Depends(get_current_user),
):
    """
    Return top mutual fund recommendations based on risk profile.
    Scores funds on 3Y/5Y returns, Sharpe, Sortino, Alpha, and expense ratio.
    """
    recs = await get_top_mutual_funds(risk_profile=risk_profile, top_n=top_n, category=category)

    return [
        MFOut(
            rank            = r.rank,
            scheme_code     = r.scheme_code,
            scheme_name     = r.scheme_name,
            category        = r.category,
            amc             = r.amc,
            nav             = r.nav,
            return_1y       = r.return_1y,
            return_3y       = r.return_3y,
            return_5y       = r.return_5y,
            sharpe_ratio    = r.sharpe_ratio,
            expense_ratio   = r.expense_ratio,
            composite_score = r.composite_score,
            risk_profile    = r.risk_profile,
            rationale       = r.rationale,
            aum             = r.aum,
        )
        for r in recs
    ]
