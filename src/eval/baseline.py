"""Naive baseline: top-K topics ranked by their *most recent* activity.

This answers the question "is our detector actually better than just
picking whatever was most popular last week?"

For each ``as_of`` checkpoint we rank topics by the mean ``count``
over the previous ``recent_weeks`` weeks, take the top-K, and check
how many of them actually grew (>=GROWTH_THRESHOLD) over the next
``horizon`` weeks. We then compare precision@K to the detector's
precision@K from ``reports/backtest/topic_precision_at_k.csv``.

Outputs:
- ``reports/backtest/baseline_top_recent.parquet`` -- per-checkpoint top-K picks
- ``reports/backtest/baseline_precision_at_k.csv`` -- precision@K for baseline
- ``reports/backtest/precision_vs_baseline.csv``  -- side-by-side comparison
- ``reports/backtest/precision_lift.csv``         -- lift = detector / baseline

Usage::

    python -m src.eval.baseline --horizon 8 --baseline 8 --recent-weeks 4
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR
from src.eval.metrics import GROWTH_THRESHOLD, ts_growth

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("baseline")

BACKTEST_DIR = REPORTS_DIR / "backtest"


def _checkpoints() -> list[pd.Timestamp]:
    """Same set of as_of dates produced by the backtest run."""
    det = pd.read_parquet(BACKTEST_DIR / "topic_detections_with_truth.parquet")
    return sorted(pd.to_datetime(det["as_of"]).dt.normalize().unique().tolist())


def baseline_topk(
    ts: pd.DataFrame,
    as_of: pd.Timestamp,
    recent_weeks: int,
    k: int,
) -> pd.DataFrame:
    """Rank topics by mean count over the last ``recent_weeks`` and return top K."""
    window_start = as_of - pd.Timedelta(weeks=recent_weeks)
    snap = ts[(ts["period"] > window_start) & (ts["period"] <= as_of)]
    if snap.empty:
        return pd.DataFrame(columns=["topic_id", "recent_mean", "rank"])
    ranked = (snap.groupby("topic_id")["count"]
                  .mean()
                  .sort_values(ascending=False)
                  .reset_index(name="recent_mean"))
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked.head(k)


def evaluate(
    horizon: int = 8,
    baseline_weeks: int = 8,
    recent_weeks: int = 4,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = pd.read_parquet(PROCESSED_DIR / "topic_timeseries.parquet")
    ts["period"] = pd.to_datetime(ts["period"])

    picks_rows: list[dict] = []
    pak_rows: list[dict] = []

    max_k = max(ks)
    for as_of in _checkpoints():
        as_of = pd.Timestamp(as_of)
        topk = baseline_topk(ts, as_of, recent_weeks, max_k)
        if topk.empty:
            continue
        # compute actual future growth for each pick
        for _, r in topk.iterrows():
            g = ts_growth(ts, int(r["topic_id"]), as_of,
                          horizon, baseline_weeks)
            picks_rows.append({
                "as_of": as_of.date(),
                "rank": int(r["rank"]),
                "topic_id": int(r["topic_id"]),
                "recent_mean": round(r["recent_mean"], 3),
                "ts_growth": g["ts_growth"],
                "actually_grew": g["actually_grew"],
            })

        for k in ks:
            sel = topk.head(k)
            if sel.empty:
                continue
            grew = 0
            n_eval = 0
            for _, r in sel.iterrows():
                g = ts_growth(ts, int(r["topic_id"]), as_of,
                              horizon, baseline_weeks)
                if pd.notna(g["ts_growth"]):
                    n_eval += 1
                    if g["actually_grew"]:
                        grew += 1
            if n_eval == 0:
                continue
            pak_rows.append({
                "as_of": as_of.date(),
                "K": k,
                "n": n_eval,
                "hits": grew,
                "precision": grew / n_eval,
            })

    picks = pd.DataFrame(picks_rows)
    base_pak = pd.DataFrame(pak_rows)

    # ---- Compare with detector precision@K --------------------------------
    det_pak_path = BACKTEST_DIR / "topic_precision_at_k.csv"
    detector_pak = pd.read_csv(det_pak_path) if det_pak_path.exists() else pd.DataFrame()

    if not detector_pak.empty and not base_pak.empty:
        d_avg = (detector_pak.groupby("K")["precision"]
                              .mean().reset_index()
                              .rename(columns={"precision": "detector_precision"}))
        b_avg = (base_pak.groupby("K")["precision"]
                          .mean().reset_index()
                          .rename(columns={"precision": "baseline_precision"}))
        compare = d_avg.merge(b_avg, on="K", how="outer").sort_values("K")
        compare["lift"] = (compare["detector_precision"] /
                           compare["baseline_precision"]).round(3)
        compare["abs_gain_pp"] = ((compare["detector_precision"] -
                                   compare["baseline_precision"]) * 100).round(1)
        compare["detector_precision"] = compare["detector_precision"].round(3)
        compare["baseline_precision"] = compare["baseline_precision"].round(3)
    else:
        compare = pd.DataFrame()

    return picks, base_pak, compare


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--baseline", dest="baseline_weeks", type=int, default=8)
    p.add_argument("--recent-weeks", type=int, default=4,
                   help="Window length for naive 'top-K by recent activity' baseline.")
    args = p.parse_args()

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    picks, base_pak, compare = evaluate(
        horizon=args.horizon,
        baseline_weeks=args.baseline_weeks,
        recent_weeks=args.recent_weeks,
    )

    if not picks.empty:
        picks_out = BACKTEST_DIR / "baseline_top_recent.parquet"
        picks.to_parquet(picks_out, index=False)
        log.info("Saved %s (%d rows)", picks_out, len(picks))

    if not base_pak.empty:
        out = BACKTEST_DIR / "baseline_precision_at_k.csv"
        base_pak.to_csv(out, index=False)
        log.info("Saved %s (%d rows)", out, len(base_pak))

    if not compare.empty:
        out = BACKTEST_DIR / "precision_vs_baseline.csv"
        compare.to_csv(out, index=False)
        log.info("Detector vs naive top-K-by-recent baseline:\n%s",
                 compare.to_string(index=False))


if __name__ == "__main__":
    main()
