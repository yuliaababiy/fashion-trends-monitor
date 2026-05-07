"""Google Trends collector via pytrends.

Fetches weekly interest-over-time for fashion-related search queries.
Output: long-format Parquet with [date, keyword, interest] suitable for the
forecasting pipeline.

Usage:
    python -m src.collect.google_trends \
        --keywords "y2k fashion" "quiet luxury" "streetwear" \
        --timeframe "today 5-y" \
        --geo "" \
        --output data/raw/google_trends.parquet
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from src.config import FASHION_KEYWORDS, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _fetch_batch_with_retry(pytrends, batch, timeframe, geo, max_retries=5, base_sleep=15.0):
    """Try a batch with exponential backoff on 429/empty responses."""
    for attempt in range(max_retries):
        try:
            pytrends.build_payload(batch, timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()
            if not df.empty:
                return df
            log.warning("Empty response (attempt %d/%d) for %s",
                        attempt + 1, max_retries, batch)
        except Exception as e:  # noqa: BLE001
            log.warning("Batch failed (attempt %d/%d): %s",
                        attempt + 1, max_retries, str(e)[:200])
        wait = base_sleep * (2 ** attempt)
        log.info("Sleeping %.1fs before retry...", wait)
        time.sleep(wait)
    return pd.DataFrame()


def fetch_trends(
    keywords: list[str],
    timeframe: str = "today 5-y",
    geo: str = "",
    sleep_between: float = 8.0,
    proxies: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch interest-over-time, one keyword per call (no anchor scaling).

    Single-keyword calls are far less likely to be 429-rate-limited and each
    series is auto-normalised to 0-100 by Google. Cross-keyword comparison of
    absolute values is therefore not meaningful — we forecast each series
    independently anyway, so this is fine.

    ``proxies`` — optional list like ``["https://user:pass@host:port", ...]``.
    pytrends rotates through them automatically, which often unblocks the
    public Trends endpoint when a single IP is rate-limited.

    Returns long-format DataFrame: [date, keyword, interest].
    """
    from pytrends.request import TrendReq

    kwargs = dict(hl="en-US", tz=0, retries=2, backoff_factor=2.0,
                  requests_args={"headers": {"Accept-Language": "en-US,en;q=0.9"}})
    if proxies:
        kwargs["proxies"] = list(proxies)
        log.info("Using %d proxy endpoint(s) (rotated by pytrends)", len(proxies))
    pytrends = TrendReq(**kwargs)

    pieces: list[pd.DataFrame] = []
    for kw in keywords:
        log.info("Fetching: %s", kw)
        df = _fetch_batch_with_retry(pytrends, [kw], timeframe, geo)
        if df.empty:
            log.warning("Skipping '%s' — no data after retries.", kw)
            continue
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        long = df.reset_index().melt(id_vars="date", var_name="keyword",
                                     value_name="interest")
        pieces.append(long)
        time.sleep(sleep_between)

    if not pieces:
        return pd.DataFrame(columns=["date", "keyword", "interest"])

    combined = pd.concat(pieces, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "keyword"]).reset_index(drop=True)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Google Trends data.")
    parser.add_argument("--keywords", nargs="+", default=FASHION_KEYWORDS,
                        help="Search queries.")
    parser.add_argument("--timeframe", default="today 5-y",
                        help="pytrends timeframe (e.g., 'today 5-y', '2020-01-01 2025-12-31').")
    parser.add_argument("--geo", default="",
                        help="Country code (e.g., 'US'). Empty = worldwide.")
    parser.add_argument("--output", type=Path,
                        default=RAW_DIR / "google_trends.parquet")
    parser.add_argument("--proxy", action="append", default=None,
                        help="Proxy URL (e.g. https://user:pass@host:port). "
                             "Pass multiple times to rotate.")
    args = parser.parse_args()

    log.info("Fetching Google Trends for %d keywords (%s, geo=%s)",
             len(args.keywords), args.timeframe, args.geo or "worldwide")
    df = fetch_trends(args.keywords, args.timeframe, args.geo,
                      proxies=args.proxy)
    if df.empty:
        log.warning("No data returned.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    log.info("Saved %d rows (%d keywords × %d weeks) to %s",
             len(df), df["keyword"].nunique(), df["date"].nunique(), args.output)


if __name__ == "__main__":
    main()
