"""Mutual Funds router — top MF recommendations by risk profile."""
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import List, Optional
from src.api.routers.auth import get_current_user, User

router = APIRouter()


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
        regex="^(CONSERVATIVE|MODERATE|AGGRESSIVE)$",
    ),
    category:  Optional[str] = Query(default=None),
    top_n:     int = Query(default=5, ge=1, le=20),
    _user:     User = Depends(get_current_user),
):
    """
    Return top mutual fund recommendations based on risk profile.
    Scores funds on 3Y/5Y returns, Sharpe, Sortino, Alpha, and expense ratio.
    """
    from src.ingestion.data_fetcher import AMFIFetcher
    from src.recommendation.engine import MutualFundRecommender, RiskProfile
    import asyncio

    fetcher = AMFIFetcher()
    nav_df  = asyncio.run(fetcher.fetch_all_navs())

    if nav_df.empty:
        return []

    # Enrich with computed returns (sample top 200 Open-Ended funds)
    open_ended = nav_df[nav_df["category"].str.contains("Open Ended", na=False)].head(200)
    enriched_rows = []

    for _, row in open_ended.iterrows():
        hist = asyncio.run(fetcher.fetch_historical_nav(str(row["scheme_code"])))
        returns = fetcher.compute_returns(hist)
        enriched_rows.append({**row.to_dict(), **returns})

    import pandas as pd
    enriched_df = pd.DataFrame(enriched_rows)

    # Fill missing risk metrics with reasonable defaults
    enriched_df["sharpe_ratio"]  = None
    enriched_df["sortino_ratio"] = None
    enriched_df["alpha"]         = None
    enriched_df["expense_ratio"] = None
    enriched_df["aum"]           = None

    recommender = MutualFundRecommender()
    recs = recommender.recommend(
        enriched_df,
        risk_profile = RiskProfile(risk_profile),
        top_n        = top_n,
        category_filter = category,
    )

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
