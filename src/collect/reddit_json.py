"""Reddit collector via public JSON endpoints (no credentials).

Reddit exposes ``/r/{sub}/{listing}.json`` and ``/search.json`` endpoints
without auth as long as the request carries a unique User-Agent and
respects ~60 requests/minute.

Usage:
    python -m src.collect.reddit_json \
        --subreddits femalefashionadvice malefashionadvice streetwear \
        --listing new --pages 5
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.config import DEFAULT_SUBREDDITS, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

UA = "fashion-trends-research/1.0 (by /u/anon-researcher)"
BASE = "https://www.reddit.com"


def fetch_listing(sub: str, listing: str = "new", limit: int = 100,
                   after: str | None = None, t: str = "year") -> dict:
    params = {"limit": limit, "raw_json": 1}
    if after:
        params["after"] = after
    if listing == "top":
        params["t"] = t
    url = f"{BASE}/r/{sub}/{listing}.json"
    r = requests.get(url, params=params,
                     headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r.json()


def post_to_row(child: dict, sub: str) -> dict:
    d = child["data"]
    return {
        "id": f"reddit_{d['id']}",
        "subreddit": sub,
        "title": d.get("title", ""),
        "selftext": d.get("selftext", "") or "",
        "author": d.get("author", ""),
        "created_utc": pd.to_datetime(d.get("created_utc", 0), unit="s", utc=True),
        "score": int(d.get("score", 0)),
        "num_comments": int(d.get("num_comments", 0)),
        "upvote_ratio": float(d.get("upvote_ratio", 1.0)),
        "url": d.get("url_overridden_by_dest") or d.get("url", ""),
        "permalink": f"https://reddit.com{d.get('permalink', '')}",
        "is_self": bool(d.get("is_self", True)),
        "over_18": bool(d.get("over_18", False)),
        "link_flair_text": d.get("link_flair_text") or "",
    }


def collect(subreddits: list[str], listing: str = "new",
            pages: int = 5, t: str = "year",
            sleep: float = 1.5) -> pd.DataFrame:
    rows: list[dict] = []
    for sub in subreddits:
        log.info("r/%s [%s]", sub, listing)
        after = None
        for page in range(pages):
            try:
                resp = fetch_listing(sub, listing=listing, after=after, t=t)
            except requests.HTTPError as e:
                log.warning("r/%s page %d: %s", sub, page, e)
                break
            data = resp.get("data", {})
            children = data.get("children", [])
            for c in children:
                if c.get("kind") == "t3":
                    rows.append(post_to_row(c, sub))
            after = data.get("after")
            if not after:
                break
            time.sleep(sleep)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset="id").reset_index(drop=True)
    return df


def merge_incremental(new_df: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Merge new rows with existing parquet, dropping duplicates by id."""
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
    p = argparse.ArgumentParser()
    p.add_argument("--subreddits", nargs="+", default=DEFAULT_SUBREDDITS)
    p.add_argument("--listing", default="new",
                   choices=["new", "hot", "top", "rising"])
    p.add_argument("--pages", type=int, default=5)
    p.add_argument("--t", default="year",
                   help="Time window for 'top' listing (hour/day/week/month/year/all).")
    p.add_argument("--output", type=Path, default=RAW_DIR / "reddit.parquet")
    p.add_argument("--incremental", action="store_true",
                   help="Append new posts to existing parquet (dedup by id).")
    args = p.parse_args()

    df = collect(args.subreddits, listing=args.listing,
                 pages=args.pages, t=args.t)
    log.info("Collected %d posts from %d subreddits",
             len(df), len(args.subreddits))
    if df.empty:
        log.warning("No posts collected.")
        return
    if args.incremental:
        df = merge_incremental(df, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    log.info("Saved %d rows to %s", len(df), args.output)


if __name__ == "__main__":
    main()
