"""Reddit OAuth-authenticated collector.

Why this exists
---------------
The public ``www.reddit.com/r/X/new.json`` endpoint is aggressively
blocked on data-center IP ranges (Azure / AWS / GCP), so the unauth
collector returns 403 on GitHub Actions runners. The authenticated
endpoint at ``oauth.reddit.com`` is *not* blocked because legitimate
API clients (script apps, personal use) authenticate there.

Setup (one-time)
----------------
1. Sign in at https://old.reddit.com/prefs/apps (must be the same
   browser session as your account).
2. Click *create another app* at the bottom.
3. Fill the form:
     - name:       fashion-trends-monitor
     - type:       **script**
     - about url:  (anything, e.g. your repo URL)
     - redirect:   http://localhost:8080
4. Click *create app*. You will see the **client id** under the app
   name (a 14-char string) and the **secret** to its right.
5. Add the following entries to ``.env`` and to GitHub Secrets:

       REDDIT_CLIENT_ID=...
       REDDIT_CLIENT_SECRET=...
       REDDIT_USERNAME=...        # your reddit account
       REDDIT_PASSWORD=...        # your reddit account password
       REDDIT_USER_AGENT=fashion-trends-monitor/1.0 by /u/<your_user>

Usage
-----
    python -m src.collect.reddit_oauth --listing new --pages 3 --incremental

The collector writes the same schema as ``reddit_json.py`` so the
downstream pipeline does not change.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import DEFAULT_SUBREDDITS, RAW_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OAUTH_URL = "https://www.reddit.com/api/v1/access_token"
API_BASE = "https://oauth.reddit.com"


def _creds() -> dict:
    load_dotenv()
    keys = ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET",
            "REDDIT_USERNAME", "REDDIT_PASSWORD", "REDDIT_USER_AGENT"]
    vals = {k: os.getenv(k, "") for k in keys}
    missing = [k for k, v in vals.items() if not v]
    if missing:
        raise RuntimeError(
            "Missing Reddit OAuth env vars: " + ", ".join(missing) +
            ".\nSee module docstring for setup instructions."
        )
    return vals


def get_token() -> tuple[str, str]:
    """Return (access_token, user_agent)."""
    c = _creds()
    auth = requests.auth.HTTPBasicAuth(c["REDDIT_CLIENT_ID"],
                                       c["REDDIT_CLIENT_SECRET"])
    data = {
        "grant_type": "password",
        "username": c["REDDIT_USERNAME"],
        "password": c["REDDIT_PASSWORD"],
    }
    headers = {"User-Agent": c["REDDIT_USER_AGENT"]}
    r = requests.post(OAUTH_URL, auth=auth, data=data,
                      headers=headers, timeout=30)
    r.raise_for_status()
    tok = r.json().get("access_token")
    if not tok:
        raise RuntimeError(f"OAuth response missing token: {r.text[:200]}")
    return tok, c["REDDIT_USER_AGENT"]


def fetch_listing(token: str, ua: str, sub: str, listing: str = "new",
                  limit: int = 100, after: str | None = None,
                  t: str = "year") -> dict:
    params = {"limit": limit, "raw_json": 1}
    if after:
        params["after"] = after
    if listing == "top":
        params["t"] = t
    url = f"{API_BASE}/r/{sub}/{listing}"
    r = requests.get(url, params=params,
                     headers={"User-Agent": ua,
                              "Authorization": f"bearer {token}"},
                     timeout=30)
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
            sleep: float = 1.0) -> pd.DataFrame:
    token, ua = get_token()
    log.info("Got OAuth token (%d chars).", len(token))

    rows: list[dict] = []
    for sub in subreddits:
        log.info("r/%s [%s]", sub, listing)
        after = None
        for page in range(pages):
            try:
                resp = fetch_listing(token, ua, sub,
                                     listing=listing, after=after, t=t)
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
    p.add_argument("--t", default="year")
    p.add_argument("--output", type=Path, default=RAW_DIR / "reddit.parquet")
    p.add_argument("--incremental", action="store_true")
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
