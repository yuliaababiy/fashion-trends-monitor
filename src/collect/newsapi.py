"""News API (newsapi.org) collector.

NOTE on the free tier:
- Articles are limited to the **last 30 days** for `/everything` endpoint.
- Up to 100 requests / day; 100 articles per request.
- For historical fashion text, prefer the Guardian collector.
- This collector is best for **incremental monthly updates** in Phase B.

Usage:
    python -m src.collect.newsapi \
        --queries "y2k fashion" "quiet luxury" \
        --output data/raw/newsapi.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from src.config import FASHION_KEYWORDS, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

API_BASE = "https://newsapi.org/v2/everything"


def get_api_key() -> str:
    load_dotenv()
    key = os.getenv("NEWSAPI_KEY")
    if not key or key.startswith("your_"):
        raise RuntimeError(
            "Missing NEWSAPI_KEY. Set it in .env (see .env.example)."
        )
    return key


def fetch_query(api_key: str, query: str, from_date: str, page_size: int = 100) -> list[dict]:
    params = {
        "q": query,
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    r = requests.get(API_BASE, params=params, timeout=30)
    if r.status_code == 426:
        # NewsAPI free plan can't go further back than 30 days.
        log.warning("NewsAPI 426: from_date too old; clamping to 30d.")
        clamped = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        params["from"] = clamped
        r = requests.get(API_BASE, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("articles", [])


def article_to_post(art: dict, query: str) -> dict:
    pub = art.get("publishedAt") or ""
    try:
        ts = pd.to_datetime(pub)
    except Exception:
        ts = pd.NaT
    title = art.get("title") or ""
    desc = art.get("description") or ""
    content = art.get("content") or ""
    body = "\n".join([x for x in (desc, content) if x])
    src = (art.get("source") or {}).get("name") or "newsapi"
    url = art.get("url") or ""
    aid = f"newsapi_{abs(hash(url))}"
    return {
        "id": aid,
        "subreddit": f"newsapi:{src}",
        "title": title,
        "selftext": body,
        "author": art.get("author") or src,
        "created_utc": ts,
        "score": 0,
        "num_comments": 0,
        "upvote_ratio": 1.0,
        "url": url,
        "permalink": url,
        "is_self": True,
        "over_18": False,
        "link_flair_text": query,
    }


def collect(queries: list[str], from_date: str, sleep_between: float = 0.5) -> pd.DataFrame:
    api_key = get_api_key()
    rows: list[dict] = []
    for q in tqdm(queries, desc="NewsAPI queries"):
        try:
            for art in fetch_query(api_key, q, from_date):
                rows.append(article_to_post(art, q))
        except requests.HTTPError as e:
            log.warning("Query '%s' failed: %s", q, e)
        time.sleep(sleep_between)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset="url").reset_index(drop=True)
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
    parser.add_argument("--queries", nargs="+", default=FASHION_KEYWORDS)
    parser.add_argument(
        "--from-date",
        default=(datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d"),
        help="ISO date. Free tier limited to ~30 days back.",
    )
    parser.add_argument("--output", type=Path, default=RAW_DIR / "newsapi.parquet")
    parser.add_argument("--incremental", action="store_true",
                        help="Append new articles to existing parquet (dedup by url).")
    args = parser.parse_args()

    df = collect(args.queries, args.from_date)
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
