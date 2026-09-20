"""
Mutual fund universe catalog: pull every scheme AMFI publishes (~14k) and
upsert NAV/category/AMC into Postgres.

    python scripts/run_mf_universe_pipeline.py

NOTE — this does NOT produce a ranked buy list. AMFI's NAV feed gives a
single point-in-time NAV per scheme with no return history, so there is no
signal to rank on (MutualFundRecommender scores on 3Y/5Y return, Sharpe,
Sortino, alpha — all None here, which would just tie every scheme at ~0).
This script populates the catalog; ranking needs a follow-up pass that
fetches historical NAV per scheme (AMFIFetcher.fetch_historical_nav) and
computes real returns for a filtered subset — much heavier (one request per
scheme), intentionally out of scope here per the "NAV-only" choice.
"""
import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select

from src.database.models import AsyncSessionFactory, MutualFund, init_db
from src.ingestion.data_fetcher import AMFIFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mf_universe_pipeline")

BATCH = 200


async def run() -> None:
    print("=" * 70)
    print("  FinanceAI — Mutual Fund Universe Catalog")
    print("=" * 70)

    await init_db()

    amfi = AMFIFetcher()
    nav_df = await amfi.fetch_all_navs()
    if nav_df.empty:
        print("  ⚠️  Could not fetch AMFI NAV data — aborting.")
        return

    nav_df = nav_df.dropna(subset=["scheme_code", "nav"])
    print(f"  Fetched {len(nav_df)} schemes with a valid NAV\n")

    start = time.time()
    upserted = 0
    rows = nav_df.to_dict("records")

    for batch_start in range(0, len(rows), BATCH):
        batch = rows[batch_start:batch_start + BATCH]
        async with AsyncSessionFactory() as db:
            codes = [r["scheme_code"] for r in batch]
            existing = (await db.execute(
                select(MutualFund).where(MutualFund.scheme_code.in_(codes))
            )).scalars().all()
            by_code = {m.scheme_code: m for m in existing}

            for r in batch:
                mf = by_code.get(r["scheme_code"])
                if not mf:
                    mf = MutualFund(scheme_code=r["scheme_code"])
                    db.add(mf)
                mf.scheme_name = r.get("scheme_name") or mf.scheme_name or ""
                mf.category = r.get("category") or mf.category or ""
                mf.amc = r.get("amc") or mf.amc or ""
                mf.nav = r.get("nav")
                upserted += 1

            await db.commit()

        if (batch_start // BATCH) % 10 == 0:
            print(f"  [{batch_start + len(batch)}/{len(rows)}] upserted so far: {upserted} "
                  f"| {time.time() - start:.0f}s elapsed")

    print("\n" + "=" * 70)
    print(f"  Done in {time.time() - start:.0f}s — {upserted} schemes upserted into `mutual_funds`")
    print("  Note: return_1y/3y/5y, sharpe/sortino/alpha, composite_score, and")
    print("  recommendation_rank are left NULL — see module docstring.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run())
