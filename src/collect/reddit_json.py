"""Reddit collector via public JSON endpoints (no credentials).

Reddit exposes ``/r/{sub}/{listing}.json`` and ``/search.json`` endpoints
without auth as long as the request carries a unique User-Agent and
respects ~60 requests/minute. Reddit aggressively blocks data-center
IP ranges (Azure / AWS / GCP) — when running on GitHub Actions you must
route requests through residential proxies.

Set ``REDDIT_PROXIES`` env var (or ``--proxies`` flag) with a
newline-, comma- or semicolon-separated list of entries in the form
``host:port:user:pass`` (Webshare format). One proxy is picked at
random per request, with rotation on failure.

Usage:
    python -m src.collect.reddit_json \\
        --subreddits femalefashionadvice malefashionadvice streetwear \\
        --listing new --pages 5
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import DEFAULT_SUBREDDITS, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

UA = "fashion-trends-research/1.0 (by /u/anon-researcher)"
BASE = "https://www.reddit.com"


def parse_proxies(raw: str | None) -> list[dict]:
    """Parse a multi-proxy string into requests-style proxy dicts.

    Accepted formats per entry (separated by newline/comma/semicolon):
        host:port:user:pass
        http://user:pass@host:port
    """
    if not raw:
        return []
    out: list[dict] = []
    for entry in re.split(r"[,\s;]+", raw.strip()):
        if not entry:
            continue
        if entry.startswith("http://") or entry.startswith("https://"):
            url = entry
        else:
            parts = entry.split(":")
            if len(parts) == 4:
                host, port, user, pwd = parts
                url = f"http://{user}:{pwd}@{host}:{port}"
            elif len(parts) == 2:
                host, port = parts
                url = f"http://{host}:{port}"
            else:
                log.warning("Skipping unrecognized proxy entry: %s", entry)
                continue
        out.append({"http": url, "https": url})
    return out


def load_proxies(cli_value: str | None) -> list[dict]:
    if cli_value:
        return parse_proxies(cli_value)
    load_dotenv()
    return parse_proxies(os.getenv("REDDIT_PROXIES"))


def fetch_listing(sub: str, listing: str = "new", limit: int = 100,
                   after: str | None = None, t: str = "year",
                   proxies: list[dict] | None = None,
                   max_retries: int = 4) -> dict:
    params = {"limit": limit, "raw_json": 1}
    if after:
        params["after"] = after
    if listing == "top":
        params["t"] = t
    url = f"{BASE}/r/{sub}/{listing}.json"
    headers = {"User-Agent": UA}

    pool = list(proxies) if proxies else [None]
    random.shuffle(pool)

    last_err: Exception | None = None
    attempts = max(max_retries, len(pool))
    for i in range(attempts):
        proxy = pool[i % len(pool)]
        try:
            r = requests.get(url, params=params, headers=headers,
                             proxies=proxy, timeout=30)
            if r.status_code in (403, 429):
                log.warning("r/%s via %s -> %s; rotating",
                            sub, _proxy_label(proxy), r.status_code)
                last_err = requests.HTTPError(f"{r.status_code} on {url}")
                time.sleep(1.0)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            log.warning("r/%s via %s -> %s; rotating",
                        sub, _proxy_label(proxy), e)
            last_err = e
            time.sleep(1.0)
            continue
    raise last_err or RuntimeError("All proxy attempts failed")


def _proxy_label(p: dict | None) -> str:
    if not p:
        return "no-proxy"
    url = p.get("http", "")
    return re.sub(r"//[^@]+@", "//***@", url)



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
            sleep: float = 1.5,
            proxies: list[dict] | None = None) -> pd.DataFrame:
    if proxies:
        log.info("Using %d proxy(ies) with rotation.", len(proxies))
    rows: list[dict] = []
    for sub in subreddits:
        log.info("r/%s [%s]", sub, listing)
        after = None
        for page in range(pages):
            try:
                resp = fetch_listing(sub, listing=listing, after=after,
                                     t=t, proxies=proxies)
            except (requests.HTTPError, requests.ConnectionError,
                    requests.Timeout) as e:
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
    p.add_argument("--proxies", default=None,
                   help="Comma/newline-separated proxy entries "
                        "(host:port:user:pass). Falls back to REDDIT_PROXIES env.")
    args = p.parse_args()

    proxies = load_proxies(args.proxies)
    df = collect(args.subreddits, listing=args.listing,
                 pages=args.pages, t=args.t, proxies=proxies)
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
