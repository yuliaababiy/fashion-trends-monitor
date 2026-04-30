"""The Guardian Open Platform collector.

Fetches fashion-tagged articles via the content API. Saves in a unified
"posts" schema compatible with the rest of the pipeline.

Usage:
    python -m src.collect.guardian \
        --from-date 2021-01-01 \
        --pages 50 \
        --output data/raw/guardian.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from src.config import RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

API_BASE = "https://content.guardianapis.com/search"


def get_api_key() -> str:
    load_dotenv()
    key = os.getenv("GUARDIAN_API_KEY")
    if not key or key.startswith("your_"):
        raise RuntimeError(
            "Missing GUARDIAN_API_KEY. Set it in .env (see .env.example)."
        )
    return key


def fetch_page(
    api_key: str,
    section: str = "fashion",
    from_date: str = "2021-01-01",
    page: int = 1,
    page_size: int = 50,
) -> dict:
    params = {
        "api-key": api_key,
        "section": section,
        "from-date": from_date,
        "page": page,
        "page-size": page_size,
        "show-fields": "trailText,bodyText,headline",
        "order-by": "newest",
    }
    r = requests.get(API_BASE, params=params, timeout=30)
    r.raise_for_status()
    return r.json()["response"]


def article_to_post(article: dict) -> dict:
    fields = article.get("fields", {})
    body = fields.get("bodyText") or fields.get("trailText", "")
    return {
        "id": f"guardian_{article['id'].replace('/', '_')}",
        "subreddit": "guardian_fashion",   # repurposed source label
        "title": article.get("webTitle", ""),
        "selftext": body,
        "author": "guardian",
        "created_utc": pd.to_datetime(article["webPublicationDate"]),
        "score": 0,
        "num_comments": 0,
        "upvote_ratio": 1.0,
        "url": article.get("webUrl", ""),
        "permalink": article.get("webUrl", ""),
        "is_self": True,
        "over_18": False,
        "link_flair_text": article.get("sectionName", ""),
    }


def collect(
    section: str = "fashion",
    from_date: str = "2021-01-01",
    max_pages: int = 50,
    page_size: int = 50,
    sleep_between: float = 0.3,
) -> pd.DataFrame:
    api_key = get_api_key()
    rows: list[dict] = []

    # First call to discover total pages.
    first = fetch_page(api_key, section, from_date, page=1, page_size=page_size)
    total_pages = min(first.get("pages", 1), max_pages)
    rows.extend(article_to_post(a) for a in first.get("results", []))
    log.info("Total pages: %d (capped at %d), %d results/page",
             first.get("pages", 1), max_pages, page_size)

    for page in tqdm(range(2, total_pages + 1), desc="Guardian pages"):
        try:
            resp = fetch_page(api_key, section, from_date, page=page, page_size=page_size)
            rows.extend(article_to_post(a) for a in resp.get("results", []))
        except requests.HTTPError as e:
            log.warning("Page %d failed: %s", page, e)
            break
        time.sleep(sleep_between)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset="id").reset_index(drop=True)
    return df


def merge_incremental(new_df: pd.DataFrame, output: Path) -> pd.DataFrame:
    if output.exists():
        try:
            old = pd.read_parquet(output)
            before = len(old)
            merged = pd.concat([old, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset="id", keep="last")
            log.info("Merged: %d existing + %d new = %d (added %d)",
                     before, len(new_df), len(merged), len(merged) - before)
            return merged.reset_index(drop=True)
        except Exception as e:
            log.warning("Could not read existing %s: %s — overwriting.", output, e)
    return new_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", default="fashion")
    parser.add_argument("--from-date", default="2021-01-01")
    parser.add_argument("--pages", type=int, default=50,
                        help="Maximum number of pages to fetch (50 articles/page).")
    parser.add_argument("--output", type=Path,
                        default=RAW_DIR / "guardian.parquet")
    parser.add_argument("--incremental", action="store_true",
                        help="Use last published date in parquet as from-date and merge.")
    args = parser.parse_args()

    from_date = args.from_date
    if args.incremental and args.output.exists():
        try:
            old = pd.read_parquet(args.output)
            if not old.empty:
                last_dt = pd.to_datetime(old["created_utc"]).max()
                # Re-fetch from a few days back to catch late updates.
                from_date = (last_dt - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
                log.info("Incremental Guardian from %s", from_date)
        except Exception as e:
            log.warning("Could not derive incremental from-date: %s", e)

    df = collect(args.section, from_date, args.pages)
    if df.empty:
        log.warning("No articles collected.")
        return

    if args.incremental:
        df = merge_incremental(df, args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    log.info("Saved %d articles to %s", len(df), args.output)
    log.info("Date range: %s -> %s",
             df["created_utc"].min(), df["created_utc"].max())


if __name__ == "__main__":
    main()
