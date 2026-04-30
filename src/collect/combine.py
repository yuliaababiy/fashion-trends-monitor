"""Combine multiple raw post-like sources (Guardian, NewsAPI, Reddit, synthetic)
into a single ``data/raw/posts.parquet`` ready for the cleaner.

Usage:
    python -m src.collect.combine \
        --inputs data/raw/guardian.parquet data/raw/newsapi.parquet \
        --output data/raw/posts.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REQUIRED_COLS = [
    "id", "subreddit", "title", "selftext", "author", "created_utc",
    "score", "num_comments", "upvote_ratio", "url", "permalink",
    "is_self", "over_18", "link_flair_text",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs", nargs="+", type=Path, required=True,
        help="Parquet files in unified post schema.",
    )
    parser.add_argument("--output", type=Path, default=RAW_DIR / "posts.parquet")
    args = parser.parse_args()

    frames = []
    for p in args.inputs:
        if not p.exists():
            log.warning("Skipping missing input: %s", p)
            continue
        df = pd.read_parquet(p)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        for c in missing:
            df[c] = pd.NA
        df = df[REQUIRED_COLS]
        log.info("%-40s %d rows", p.name, len(df))
        frames.append(df)

    if not frames:
        raise RuntimeError("No input files loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="id").reset_index(drop=True)
    combined["created_utc"] = pd.to_datetime(combined["created_utc"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["created_utc"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    log.info("Saved %d combined rows to %s", len(combined), args.output)


if __name__ == "__main__":
    main()
