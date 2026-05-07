"""Backfill Google Trends for all FASHION_KEYWORDS, merging with existing.

Reads existing data/raw/google_trends.parquet, fetches only missing keywords
at the requested timeframe, then writes the merged result back.

Usage:
    python scripts/backfill_google_trends.py --timeframe "today 5-y"
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.collect.google_trends import fetch_trends
from src.config import FASHION_KEYWORDS, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default="today 5-y")
    parser.add_argument("--geo", default="")
    parser.add_argument("--output", type=Path, default=RAW_DIR / "google_trends.parquet")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch all keywords even if already present.")
    parser.add_argument("--keywords", nargs="*", default=None,
                        help="Override keyword list (default = FASHION_KEYWORDS).")
    parser.add_argument("--proxy", action="append", default=None,
                        help="Proxy URL (https://user:pass@host:port). "
                             "Pass multiple times to rotate.")
    args = parser.parse_args()

    target_keywords = args.keywords or FASHION_KEYWORDS

    existing = pd.DataFrame(columns=["date", "keyword", "interest"])
    if args.output.exists():
        existing = pd.read_parquet(args.output)
        log.info("Existing parquet: %d rows, %d keywords (%s)",
                 len(existing), existing["keyword"].nunique(),
                 sorted(existing["keyword"].unique()))

    if args.force:
        missing = list(target_keywords)
    else:
        present = set(existing["keyword"].unique())
        missing = [k for k in target_keywords if k not in present]

    if not missing:
        log.info("Nothing to fetch — all keywords already present.")
        return

    log.info("Fetching %d new keywords: %s", len(missing), missing)
    new_df = fetch_trends(missing, args.timeframe, args.geo,
                          proxies=args.proxy)

    if new_df.empty:
        log.warning("Got no data; nothing to merge.")
        return

    if args.force:
        # Drop any existing rows for the keywords we re-fetched.
        existing = existing[~existing["keyword"].isin(missing)]

    merged = pd.concat([existing, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "keyword"]).reset_index(drop=True)
    merged["date"] = pd.to_datetime(merged["date"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    log.info("Saved %d rows (%d keywords) to %s",
             len(merged), merged["keyword"].nunique(), args.output)


if __name__ == "__main__":
    main()
