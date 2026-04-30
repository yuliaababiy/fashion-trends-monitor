"""Mastodon collector via public hashtag timeline.

Public ``/api/v1/timelines/tag/{tag}`` endpoint requires no auth on most
instances. We rotate across a few large fashion-friendly instances and
combine results.

Usage:
    python -m src.collect.mastodon --tags fashion streetwear y2k cottagecore
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

from src.config import RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

INSTANCES = [
    "https://mastodon.social",
    "https://mastodon.online",
    "https://mas.to",
    "https://fosstodon.org",
]
UA = "fashion-trends-research/1.0"

DEFAULT_TAGS = [
    "fashion", "streetwear", "y2k", "cottagecore", "darkacademia",
    "quietluxury", "ootd", "vintage", "thrifting", "sustainablefashion",
]


def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s or "").strip()


def fetch_tag(instance: str, tag: str, max_id: str | None = None,
              limit: int = 40) -> list[dict]:
    url = f"{instance}/api/v1/timelines/tag/{tag}"
    params = {"limit": min(limit, 40)}
    if max_id:
        params["max_id"] = max_id
    r = requests.get(url, params=params,
                     headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r.json()


def status_to_row(s: dict, tag: str) -> dict:
    text = strip_html(s.get("content", ""))
    acct = s.get("account", {}).get("acct", "")
    return {
        "id": f"masto_{s.get('id','')}",
        "subreddit": f"mastodon:{tag}",
        "title": text[:120],
        "selftext": text,
        "author": acct,
        "created_utc": pd.to_datetime(s.get("created_at"), utc=True, errors="coerce"),
        "score": int(s.get("favourites_count", 0)),
        "num_comments": int(s.get("replies_count", 0)),
        "upvote_ratio": 1.0,
        "url": s.get("url", ""),
        "permalink": s.get("url", ""),
        "is_self": True,
        "over_18": bool(s.get("sensitive", False)),
        "link_flair_text": tag,
    }


def collect(tags: list[str], pages: int = 4,
            sleep: float = 1.0) -> pd.DataFrame:
    rows: list[dict] = []
    for tag in tags:
        for instance in INSTANCES:
            log.info("%s #%s", instance.split("//")[1], tag)
            max_id = None
            for page in range(pages):
                try:
                    items = fetch_tag(instance, tag, max_id=max_id)
                except requests.HTTPError as e:
                    log.warning("%s #%s page %d: %s", instance, tag, page, e)
                    break
                if not items:
                    break
                for s in items:
                    rows.append(status_to_row(s, tag))
                max_id = items[-1].get("id")
                time.sleep(sleep)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["created_utc"])
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
    p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    p.add_argument("--pages", type=int, default=4)
    p.add_argument("--output", type=Path, default=RAW_DIR / "mastodon.parquet")
    p.add_argument("--incremental", action="store_true",
                   help="Append new statuses to existing parquet (dedup by id).")
    args = p.parse_args()

    df = collect(args.tags, pages=args.pages)
    log.info("Collected %d statuses for %d tags", len(df), len(args.tags))
    if df.empty:
        log.warning("No statuses collected.")
        return
    if args.incremental:
        df = merge_incremental(df, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    log.info("Saved %d rows to %s", len(df), args.output)


if __name__ == "__main__":
    main()
