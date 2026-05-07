"""Compute validation metrics for the backtest results.

Inputs (produced by ``src.eval.backtest``):
- ``reports/backtest/<as_of>/emerging_topics.csv``
- ``reports/backtest/<as_of>/emerging_trends.csv``
- ``reports/backtest/<as_of>/spike_terms.csv``
- ``reports/backtest/<as_of>/forecast_metrics.csv``
- ``reports/backtest/<as_of>/trends_metrics.csv``

Ground truth: ``data/raw/google_trends.parquet`` (full series, weekly
interest 0-100 per keyword). For each detection at ``as_of``, we
compute the actual growth observed in the *next* ``horizon`` weeks
relative to the previous ``baseline`` weeks of GT interest.

Outputs:
- ``reports/backtest/summary.csv`` - one-row headline metrics
- ``reports/backtest/precision_at_k.csv`` - precision over K=1..20
- ``reports/backtest/forecast_models_comparison.csv`` - mean MAE/RMSE/MAPE/sMAPE
  by model across checkpoints (topic series + trends series)
- ``reports/backtest/detections_with_truth.parquet`` - every GT-trend detection
  joined with its observed future growth (for case-study selection)

Usage::

    python -m src.eval.metrics --horizon 8 --baseline 8
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("metrics")

BACKTEST_DIR = REPORTS_DIR / "backtest"

GROWTH_THRESHOLD = 0.15  # +15% is "actually grew"


# ---------------------------------------------------------------------------
def load_gt(path: Path = RAW_DIR / "google_trends.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_topic_ts(path: Path = PROCESSED_DIR / "topic_timeseries.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["period"] = pd.to_datetime(df["period"])
    return df


def ts_growth(
    ts: pd.DataFrame,
    topic_id: int,
    as_of: pd.Timestamp,
    horizon_weeks: int,
    baseline_weeks: int,
    value_col: str = "count",
) -> dict:
    """Like ``gt_growth`` but for our internal topic timeseries."""
    sub = ts[ts["topic_id"] == topic_id].copy()
    if sub.empty:
        return {"ts_baseline": float("nan"), "ts_future": float("nan"),
                "ts_growth": float("nan"), "actually_grew": False}
    sub = sub.sort_values("period")
    past = sub[(sub["period"] <= as_of)
               & (sub["period"] >= as_of - pd.Timedelta(weeks=baseline_weeks))]
    fut = sub[(sub["period"] > as_of)
              & (sub["period"] <= as_of + pd.Timedelta(weeks=horizon_weeks))]
    if past.empty or fut.empty:
        return {"ts_baseline": float("nan"), "ts_future": float("nan"),
                "ts_growth": float("nan"), "actually_grew": False}
    base = past[value_col].mean()
    fmean = fut[value_col].mean()
    growth = (fmean - base) / base if base else float("nan")
    actually = growth >= GROWTH_THRESHOLD if pd.notna(growth) else False
    return {"ts_baseline": float(base), "ts_future": float(fmean),
            "ts_growth": float(growth) if pd.notna(growth) else float("nan"),
            "actually_grew": bool(actually)}


def gt_growth(
    gt: pd.DataFrame,
    keyword: str,
    as_of: pd.Timestamp,
    horizon_weeks: int,
    baseline_weeks: int,
) -> dict:
    """Compute observed growth of GT interest for one keyword around ``as_of``.

    Returns a dict with mean values of the previous ``baseline_weeks`` and
    next ``horizon_weeks``, the ratio, and a Boolean ``actually_grew`` flag.
    """
    sub = gt[gt["keyword"] == keyword].copy()
    if sub.empty:
        return {"gt_baseline": float("nan"), "gt_future": float("nan"),
                "gt_growth": float("nan"), "actually_grew": False,
                "gt_peak_date": pd.NaT}

    sub = sub.sort_values("date")
    past_window = sub[(sub["date"] <= as_of)
                      & (sub["date"] >= as_of - pd.Timedelta(weeks=baseline_weeks))]
    future_window = sub[(sub["date"] > as_of)
                        & (sub["date"] <= as_of + pd.Timedelta(weeks=horizon_weeks))]
    if past_window.empty or future_window.empty:
        return {"gt_baseline": float("nan"), "gt_future": float("nan"),
                "gt_growth": float("nan"), "actually_grew": False,
                "gt_peak_date": pd.NaT}

    base_mean = past_window["interest"].mean()
    fut_mean = future_window["interest"].mean()
    growth = (fut_mean - base_mean) / base_mean if base_mean else float("nan")
    actually = growth >= GROWTH_THRESHOLD if pd.notna(growth) else False
    peak_idx = future_window["interest"].idxmax()
    peak_date = future_window.loc[peak_idx, "date"]
    return {"gt_baseline": float(base_mean),
            "gt_future": float(fut_mean),
            "gt_growth": float(growth) if pd.notna(growth) else float("nan"),
            "actually_grew": bool(actually),
            "gt_peak_date": peak_date}


# ---------------------------------------------------------------------------
def evaluate_topics(
    horizon: int,
    baseline: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score topic detections vs actual future topic counts."""
    pq = BACKTEST_DIR / "all_emerging_topics.parquet"
    if not pq.exists():
        log.warning("Missing %s; run backtest first.", pq)
        return pd.DataFrame(), pd.DataFrame()

    det = pd.read_parquet(pq)
    det["as_of"] = pd.to_datetime(det["as_of"])
    ts = load_topic_ts()

    rows: list[dict] = []
    for _, r in det.iterrows():
        truth = ts_growth(ts, int(r["topic_id"]), r["as_of"], horizon, baseline)
        rows.append({
            "as_of": r["as_of"],
            "topic_id": int(r["topic_id"]),
            "keywords": r.get("keywords"),
            "status": r.get("status"),
            "momentum_pct": r.get("momentum_pct"),
            "forecast_pct": r.get("forecast_pct"),
            **truth,
        })
    out = pd.DataFrame(rows)
    out_path = BACKTEST_DIR / "topic_detections_with_truth.parquet"
    out.to_parquet(out_path, index=False)
    log.info("Saved %s (%d rows)", out_path, len(out))

    # Precision@K on Rising topics ranked by forecast_pct/momentum.
    pak: list[dict] = []
    for as_of, grp in out.groupby("as_of"):
        rising = grp[grp["status"] == "Rising"].copy()
        if rising.empty:
            continue
        rising["score"] = rising["forecast_pct"].fillna(rising["momentum_pct"])
        rising = rising.sort_values("score", ascending=False)
        for k in (1, 3, 5, 10):
            top = rising.head(k)
            if top.empty:
                continue
            valid = top.dropna(subset=["actually_grew"])
            if valid.empty:
                continue
            hits = valid["actually_grew"].sum()
            pak.append({"as_of": as_of, "K": k, "n": int(len(valid)),
                        "hits": int(hits), "precision": hits / len(valid)})
    pak_df = pd.DataFrame(pak)
    if not pak_df.empty:
        pak_df.to_csv(BACKTEST_DIR / "topic_precision_at_k.csv", index=False)
        log.info("Saved topic_precision_at_k.csv (%d rows)", len(pak_df))
    return out, pak_df


# ---------------------------------------------------------------------------
def evaluate_trends(
    horizon: int,
    baseline: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score GT-keyword detections against actual observed growth.

    Returns (detections_with_truth, precision_at_k).
    """
    pq = BACKTEST_DIR / "all_emerging_trends.parquet"
    if not pq.exists():
        log.warning("Missing %s; run backtest first.", pq)
        return pd.DataFrame(), pd.DataFrame()

    det = pd.read_parquet(pq)
    det["as_of"] = pd.to_datetime(det["as_of"])
    gt = load_gt()

    rows: list[dict] = []
    for _, r in det.iterrows():
        truth = gt_growth(gt, r["keyword"], r["as_of"], horizon, baseline)
        rows.append({
            "as_of": r["as_of"],
            "keyword": r["keyword"],
            "status": r.get("status"),
            "momentum_pct": r.get("momentum_pct"),
            "forecast_pct": r.get("forecast_pct"),
            **truth,
        })

    out = pd.DataFrame(rows)
    out_path = BACKTEST_DIR / "detections_with_truth.parquet"
    out.to_parquet(out_path, index=False)
    log.info("Saved %s (%d rows)", out_path, len(out))

    # Precision@K computed on Rising detections sorted by forecast_pct.
    pak: list[dict] = []
    for as_of, grp in out.groupby("as_of"):
        rising = grp[grp["status"] == "Rising"].copy()
        if rising.empty:
            continue
        rising["score"] = rising["forecast_pct"].fillna(rising["momentum_pct"])
        rising = rising.sort_values("score", ascending=False)
        for k in (1, 3, 5, 10):
            top = rising.head(k)
            if top.empty:
                continue
            hits = top["actually_grew"].sum()
            pak.append({"as_of": as_of, "K": k, "n": len(top), "hits": int(hits),
                        "precision": hits / len(top)})
    pak_df = pd.DataFrame(pak)
    if not pak_df.empty:
        pak_df.to_csv(BACKTEST_DIR / "precision_at_k.csv", index=False)
        log.info("Saved precision_at_k.csv (%d rows)", len(pak_df))

    return out, pak_df


# ---------------------------------------------------------------------------
def aggregate_forecast_metrics() -> pd.DataFrame:
    """Combine per-checkpoint forecast_metrics.csv files into one comparison."""
    rows: list[pd.DataFrame] = []
    for sub in sorted(BACKTEST_DIR.iterdir()):
        if not sub.is_dir():
            continue
        for fname in ("forecast_metrics.csv", "trends_metrics.csv"):
            p = sub / fname
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
            except Exception as e:  # noqa: BLE001
                log.warning("Could not read %s: %s", p, e)
                continue
            df["as_of"] = pd.to_datetime(sub.name)
            df["series_kind"] = "topics" if fname == "forecast_metrics.csv" else "trends"
            rows.append(df)
    if not rows:
        log.warning("No per-checkpoint forecast_metrics found.")
        return pd.DataFrame()

    big = pd.concat(rows, ignore_index=True)
    cols = [c for c in ("MAE", "RMSE", "MAPE", "sMAPE") if c in big.columns]
    summary = (big.groupby(["series_kind", "model"])[cols]
                  .mean().round(3).reset_index()
                  .sort_values(["series_kind", "MAE"]))
    summary.to_csv(BACKTEST_DIR / "forecast_models_comparison.csv", index=False)
    log.info("Saved forecast_models_comparison.csv:\n%s", summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
def headline_summary(
    det: pd.DataFrame,
    pak: pd.DataFrame,
) -> pd.DataFrame:
    """Build a single-row dashboard of top-line numbers."""
    parts = {}
    if not det.empty:
        rising = det[det["status"] == "Rising"]
        valid = rising.dropna(subset=["actually_grew"])
        if not valid.empty:
            parts["rising_total"] = int(len(valid))
            parts["rising_hits"] = int(valid["actually_grew"].sum())
            parts["rising_precision"] = round(valid["actually_grew"].mean(), 3)
        # Recall: among all keyword/as_of pairs where GT actually grew, how
        # many did we flag Rising?
        all_pairs = det.dropna(subset=["actually_grew"])
        actual_growers = all_pairs[all_pairs["actually_grew"]]
        if not actual_growers.empty:
            caught = actual_growers["status"] == "Rising"
            parts["recall"] = round(float(caught.mean()), 3)
            parts["actual_growers_n"] = int(len(actual_growers))
    if not pak.empty:
        for k in (1, 3, 5, 10):
            sub = pak[pak["K"] == k]
            if not sub.empty:
                parts[f"precision_at_{k}"] = round(float(sub["precision"].mean()), 3)

    summary = pd.DataFrame([parts])
    out = BACKTEST_DIR / "summary.csv"
    summary.to_csv(out, index=False)
    log.info("Saved %s:\n%s", out, summary.T.to_string(header=False))
    return summary


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=8,
                   help="Future window in weeks for growth measurement.")
    p.add_argument("--baseline", type=int, default=8,
                   help="Past window in weeks for baseline measurement.")
    args = p.parse_args()

    det_topics, pak_topics = evaluate_topics(args.horizon, args.baseline)
    det_trends, pak_trends = evaluate_trends(args.horizon, args.baseline)
    aggregate_forecast_metrics()

    # Combine both ground-truth channels for the headline summary.
    combined = []
    if not det_topics.empty:
        d = det_topics.copy()
        d["channel"] = "topics"
        d["label"] = d["keywords"].fillna("").str.slice(0, 40)
        combined.append(d)
    if not det_trends.empty:
        d = det_trends.copy()
        d["channel"] = "trends"
        d["label"] = d["keyword"]
        combined.append(d)
    det_all = (pd.concat(combined, ignore_index=True)
               if combined else pd.DataFrame())

    pak_all = []
    if not pak_topics.empty:
        p1 = pak_topics.copy(); p1["channel"] = "topics"; pak_all.append(p1)
    if not pak_trends.empty:
        p2 = pak_trends.copy(); p2["channel"] = "trends"; pak_all.append(p2)
    pak_combined = (pd.concat(pak_all, ignore_index=True)
                    if pak_all else pd.DataFrame())
    if not pak_combined.empty:
        pak_combined.to_csv(BACKTEST_DIR / "precision_at_k.csv", index=False)

    headline_summary(det_all, pak_combined)


if __name__ == "__main__":
    main()
