"""Lead-time analysis for the backtest detections.

For every Rising detection at ``as_of`` we look at the future window
``[as_of, as_of + horizon weeks]`` in the underlying time series and
ask:

- When does the **peak** of the trend occur within that window?
- How many weeks **before** the peak did the system raise the flag?

Outputs:
- ``reports/backtest/lead_time_topics.parquet``
- ``reports/backtest/lead_time_summary.csv`` (mean / median / hist)

Usage::

    python -m src.eval.lead_time --horizon 8
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lead_time")

BACKTEST_DIR = REPORTS_DIR / "backtest"


def _peak_offset(future: pd.Series) -> tuple[int, float, float]:
    """Return (weeks_to_peak, peak_value, mean_value).

    weeks_to_peak is 1-based: 1 = peak is the first week after as_of.
    """
    if future.empty or future.isna().all() or future.max() <= 0:
        return -1, float("nan"), float("nan")
    idx = int(future.values.argmax())
    return idx + 1, float(future.iloc[idx]), float(future.mean())


def topic_lead_times(horizon: int = 8) -> pd.DataFrame:
    det = pd.read_parquet(BACKTEST_DIR / "topic_detections_with_truth.parquet")
    det["as_of"] = pd.to_datetime(det["as_of"])
    rising = det[det["status"] == "Rising"].copy()

    ts = pd.read_parquet(PROCESSED_DIR / "topic_timeseries.parquet")
    ts["period"] = pd.to_datetime(ts["period"])

    rows: list[dict] = []
    for _, r in rising.iterrows():
        sub = ts[ts["topic_id"] == r["topic_id"]].sort_values("period")
        future = sub[(sub["period"] > r["as_of"]) &
                     (sub["period"] <= r["as_of"] + pd.Timedelta(weeks=horizon))]["count"]
        wk, peak, mean_v = _peak_offset(future)
        rows.append({
            "as_of": r["as_of"].date(),
            "topic_id": int(r["topic_id"]),
            "keywords": (str(r.get("keywords") or ""))[:60],
            "ts_growth": r.get("ts_growth"),
            "actually_grew": r.get("actually_grew"),
            "weeks_to_peak": wk,
            "peak_value": peak,
            "future_mean": mean_v,
        })
    return pd.DataFrame(rows)


def trend_lead_times(horizon: int = 8) -> pd.DataFrame:
    p = BACKTEST_DIR / "detections_with_truth.parquet"
    if not p.exists():
        return pd.DataFrame()
    det = pd.read_parquet(p)
    det["as_of"] = pd.to_datetime(det["as_of"])
    rising = det[det["status"] == "Rising"].copy()
    if rising.empty:
        return pd.DataFrame()

    gt = pd.read_parquet(RAW_DIR / "google_trends.parquet")
    gt["date"] = pd.to_datetime(gt["date"])

    rows: list[dict] = []
    for _, r in rising.iterrows():
        sub = gt[gt["keyword"] == r["keyword"]].sort_values("date")
        future = sub[(sub["date"] > r["as_of"]) &
                     (sub["date"] <= r["as_of"] + pd.Timedelta(weeks=horizon))]["interest"]
        wk, peak, mean_v = _peak_offset(future)
        rows.append({
            "as_of": r["as_of"].date(),
            "keyword": r["keyword"],
            "gt_growth": r.get("gt_growth"),
            "actually_grew": r.get("actually_grew"),
            "weeks_to_peak": wk,
            "peak_value": peak,
            "future_mean": mean_v,
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=8)
    args = p.parse_args()

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    topics = topic_lead_times(args.horizon)
    trends = trend_lead_times(args.horizon)

    topics_out = BACKTEST_DIR / "lead_time_topics.parquet"
    topics.to_parquet(topics_out, index=False)
    log.info("Saved %s (%d rows)", topics_out, len(topics))

    if not trends.empty:
        trends_out = BACKTEST_DIR / "lead_time_trends.parquet"
        trends.to_parquet(trends_out, index=False)
        log.info("Saved %s (%d rows)", trends_out, len(trends))

    # ---- Summary statistics over hits only --------------------------------
    rows: list[dict] = []
    for kind, df in [("topics", topics), ("trends", trends)]:
        if df.empty:
            continue
        hits = df[(df["actually_grew"] == True) & (df["weeks_to_peak"] > 0)]
        if hits.empty:
            continue
        rows.append({
            "channel": kind,
            "n_rising_hits": int(len(hits)),
            "mean_weeks_to_peak": round(hits["weeks_to_peak"].mean(), 2),
            "median_weeks_to_peak": float(hits["weeks_to_peak"].median()),
            "p25_weeks": float(hits["weeks_to_peak"].quantile(0.25)),
            "p75_weeks": float(hits["weeks_to_peak"].quantile(0.75)),
            "min_weeks": int(hits["weeks_to_peak"].min()),
            "max_weeks": int(hits["weeks_to_peak"].max()),
            "early_warning_pct": round(
                100.0 * (hits["weeks_to_peak"] >= 4).mean(), 1
            ),
        })
    summary = pd.DataFrame(rows)
    out = BACKTEST_DIR / "lead_time_summary.csv"
    summary.to_csv(out, index=False)
    log.info("Saved %s:\n%s", out, summary.to_string(index=False))

    # Histogram for topics channel
    if not topics.empty:
        hits = topics[(topics["actually_grew"] == True) &
                      (topics["weeks_to_peak"] > 0)]
        if not hits.empty:
            hist = (hits.groupby("weeks_to_peak").size()
                        .reindex(range(1, args.horizon + 1), fill_value=0)
                        .reset_index(name="n_hits"))
            hist["share"] = (hist["n_hits"] / hist["n_hits"].sum()).round(3)
            hist_out = BACKTEST_DIR / "lead_time_hist_topics.csv"
            hist.to_csv(hist_out, index=False)
            log.info("Saved %s:\n%s", hist_out,
                     hist.to_string(index=False))


if __name__ == "__main__":
    main()
