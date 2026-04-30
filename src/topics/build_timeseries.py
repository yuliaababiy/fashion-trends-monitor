"""Build per-topic time series from doc-topic assignments.

Aggregates document counts by (topic_id, period) where period defaults to weekly.
Adds engagement-weighted variants (sum of post score) for richer signals.

Usage:
    python -m src.topics.build_timeseries \
        --input data/processed/bertopic_doc_topics.parquet \
        --freq W \
        --output data/processed/topic_timeseries.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, TIME_FREQ

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def build_topic_timeseries(
    df: pd.DataFrame,
    freq: str = TIME_FREQ,
    date_col: str = "date",
    topic_col: str = "topic_id",
    score_col: str = "score",
    drop_outliers: bool = True,
) -> pd.DataFrame:
    """Return long-format DF: [period, topic_id, count, score_sum]."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if drop_outliers:
        df = df[df[topic_col] >= 0]

    df["period"] = df[date_col].dt.to_period(freq).dt.start_time

    agg = (
        df.groupby(["period", topic_col])
          .agg(count=(topic_col, "size"),
               score_sum=(score_col, "sum"),
               score_mean=(score_col, "mean"))
          .reset_index()
    )

    # Make a complete (topic, period) grid (filling gaps with 0).
    # Build the date range from the existing period values so the calendar
    # aligns exactly with `to_period(freq).dt.start_time` (e.g. Monday for W).
    period_start = agg["period"].min()
    period_end = agg["period"].max()
    full_periods = pd.period_range(period_start, period_end, freq=freq).to_timestamp()
    periods = full_periods
    topics = sorted(agg[topic_col].unique())
    grid = pd.MultiIndex.from_product(
        [periods, topics], names=["period", topic_col]
    ).to_frame(index=False)
    out = grid.merge(agg, on=["period", topic_col], how="left").fillna(0)
    out["count"] = out["count"].astype(int)
    return out.sort_values([topic_col, "period"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Doc-topic parquet (LDA or BERTopic output).")
    parser.add_argument("--freq", default=TIME_FREQ, help="Pandas freq alias (D, W, M).")
    parser.add_argument("--output", type=Path,
                        default=PROCESSED_DIR / "topic_timeseries.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    ts = build_topic_timeseries(df, freq=args.freq)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ts.to_parquet(args.output, index=False)
    log.info("Saved %d rows (%d topics) to %s",
             len(ts), ts["topic_id"].nunique(), args.output)


if __name__ == "__main__":
    main()
