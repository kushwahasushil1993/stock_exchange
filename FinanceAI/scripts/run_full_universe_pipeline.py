"""
Full-universe pipeline: fetch every NSE-listed equity (~2,300 symbols) and
every F&O-eligible stock among them (~210), run the prediction engine on
each, and persist buy/sell signals to Postgres.

    python scripts/run_full_universe_pipeline.py                 # full run
    python scripts/run_full_universe_pipeline.py --limit 50       # smoke test
    python scripts/run_full_universe_pipeline.py --workers 16     # more concurrency

The actual scan logic lives in src/pipelines/universe_scan.py, shared with
the API trigger (POST /api/v1/predictions/run-universe) — this script is
just a CLI wrapper that prints progress to stdout.
"""
import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.universe_scan import run_universe_scan

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _print_progress(event: dict) -> None:
    kind = event["event"]
    if kind == "started":
        print("=" * 70)
        print("  FinanceAI — Full NSE Universe Scan")
        print("=" * 70)
        print(f"  NSE equities: {event['equities']} | F&O-eligible: {event['fno_eligible']}")
        print(f"  Scanning {event['scanning']} symbols with {event['workers']} workers…\n")
    elif kind == "progress":
        print(f"  [{event['scanned']}/{event['total']}] ok={event['ok']} failed={event['failed']} "
              f"| BUY={event['buy']} SELL={event['sell']} HOLD={event['hold']} | {event['elapsed_s']}s elapsed")
    elif kind == "completed":
        print("\n" + "=" * 70)
        print(f"  Done in {event['elapsed_s']}s — {event['ok']} predicted, {event['failed']} skipped/failed")
        print(f"  BUY={event['buy']}  SELL={event['sell']}  HOLD={event['hold']}")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N symbols (smoke test)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent fetch threads")
    parser.add_argument("--universe", choices=["all", "fno"], default="all",
                         help="'all' = every NSE equity, 'fno' = only F&O-eligible stocks")
    args = parser.parse_args()
    asyncio.run(run_universe_scan(args.limit, args.workers, args.universe, on_progress=_print_progress))
