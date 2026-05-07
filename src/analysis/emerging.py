"""Compute emerging trend rankings.

For each topic and Google Trends keyword, compute:
  * recent_mean    - average of last 8 weeks
  * baseline_mean  - average of weeks [-26, -8] (earlier history)
  * momentum_pct   - (recent - baseline) / baseline * 100
  * forecast_mean  - mean of best-model forecast (LSTM if present)
  * forecast_pct   - (forecast - recent) / recent * 100
  * status         - Rising / Stable / Declining

Outputs:
    reports/metrics/emerging_topics.csv
    reports/metrics/emerging_trends.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import METRICS_DIR, PROCESSED_DIR, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RECENT_W = 8
BASELINE_W = 26  # weeks
RISE_THR = 15.0  # % growth to be classified Rising
DECL_THR = -15.0


def _classify(pct: float) -> str:
    if pd.isna(pct):
        return "Unknown"
    if pct >= RISE_THR:
        return "Rising"
    if pct <= DECL_THR:
        return "Declining"
    return "Stable"


def _pick_model(fc_df: pd.DataFrame) -> str:
    if fc_df.empty:
        return ""
    preferred = ["LSTM", "XGBoost-lag8", "SARIMA(1,1,1)(1,0,1,52)",
                 "ARIMA(1,1,1)", "Prophet", "MA(4)", "Naive"]
    avail = set(fc_df["model"].unique())
    for m in preferred:
        if m in avail:
            return m
    return sorted(avail)[0]


def _row_stats(history: pd.Series, future: pd.Series | None) -> dict:
    history = history.dropna()
    n = len(history)
    recent = history.tail(RECENT_W).mean() if n >= 1 else float("nan")
    if n >= RECENT_W + BASELINE_W:
        baseline = history.iloc[-(RECENT_W + BASELINE_W):-RECENT_W].mean()
    elif n > RECENT_W:
        baseline = history.iloc[:-RECENT_W].mean()
    else:
        baseline = history.mean()

    momentum_pct = ((recent - baseline) / baseline * 100.0) if baseline else 0.0

    fc_mean = float("nan")
    fc_pct = float("nan")
    if future is not None and len(future):
        fc_mean = future.mean()
        if recent and recent > 0:
            fc_pct = (fc_mean - recent) / recent * 100.0

    return dict(
        recent_mean=round(recent, 2),
        baseline_mean=round(baseline, 2),
        momentum_pct=round(momentum_pct, 1),
        forecast_mean=round(fc_mean, 2) if pd.notna(fc_mean) else None,
        forecast_pct=round(fc_pct, 1) if pd.notna(fc_pct) else None,
    )


def emerging_topics(
    ts_path: Path | None = None,
    fc_path: Path | None = None,
    topics_path: Path | None = None,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    ts_path = Path(ts_path) if ts_path else PROCESSED_DIR / "topic_timeseries.parquet"
    fc_path = Path(fc_path) if fc_path else PROCESSED_DIR / "forecasts.parquet"
    topics_path = Path(topics_path) if topics_path else PROCESSED_DIR / "lda_topics.csv"

    if not ts_path.exists():
        log.warning("No %s", ts_path)
        return pd.DataFrame()

    ts = pd.read_parquet(ts_path)
    if as_of is not None:
        ts = ts[pd.to_datetime(ts["period"]) <= pd.to_datetime(as_of)].copy()
    fc = pd.read_parquet(fc_path) if fc_path.exists() else pd.DataFrame()
    topics = pd.read_csv(topics_path) if topics_path.exists() else pd.DataFrame()

    chosen_model = _pick_model(fc)
    log.info("Using forecast model: %s", chosen_model)
    fc_best = fc[fc["model"] == chosen_model] if chosen_model else pd.DataFrame()

    rows = []
    for tid, grp in ts.groupby("topic_id"):
        grp = grp.sort_values("period")
        future = None
        if not fc_best.empty:
            sub = fc_best[fc_best["topic_id"] == tid]
            if not sub.empty:
                future = sub["y_pred"]
        stats = _row_stats(grp["count"], future)

        kw = ""
        if not topics.empty and "topic_id" in topics.columns:
            r = topics[topics["topic_id"] == tid]
            if not r.empty:
                kw = str(r.iloc[0].get("keywords", ""))[:120]

        rows.append({
            "topic_id": int(tid),
            "keywords": kw,
            **stats,
            "status": _classify(stats["forecast_pct"]
                                if stats["forecast_pct"] is not None
                                else stats["momentum_pct"]),
            "model": chosen_model,
        })

    out = pd.DataFrame(rows).sort_values(
        ["forecast_pct", "momentum_pct"], ascending=False, na_position="last"
    ).reset_index(drop=True)
    return out


def emerging_trends(
    ts_path: Path | None = None,
    fc_path: Path | None = None,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    ts_path = Path(ts_path) if ts_path else RAW_DIR / "google_trends.parquet"
    fc_path = Path(fc_path) if fc_path else PROCESSED_DIR / "trends_forecasts.parquet"

    if not ts_path.exists():
        log.warning("No google_trends.parquet")
        return pd.DataFrame()

    ts = pd.read_parquet(ts_path)
    if as_of is not None:
        ts = ts[pd.to_datetime(ts["date"]) <= pd.to_datetime(as_of)].copy()
    fc = pd.read_parquet(fc_path) if fc_path.exists() else pd.DataFrame()

    chosen_model = _pick_model(fc)
    log.info("Using forecast model for trends: %s", chosen_model)
    fc_best = fc[fc["model"] == chosen_model] if chosen_model else pd.DataFrame()

    rows = []
    for kw, grp in ts.groupby("keyword"):
        grp = grp.sort_values("date")
        future = None
        if not fc_best.empty:
            sub = fc_best[fc_best["keyword"] == kw]
            if not sub.empty:
                future = sub["y_pred"]
        stats = _row_stats(grp["interest"], future)
        rows.append({
            "keyword": kw,
            **stats,
            "status": _classify(stats["forecast_pct"]
                                if stats["forecast_pct"] is not None
                                else stats["momentum_pct"]),
            "model": chosen_model,
        })

    out = pd.DataFrame(rows).sort_values(
        ["forecast_pct", "momentum_pct"], ascending=False, na_position="last"
    ).reset_index(drop=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ts-topics", type=Path,
                   default=PROCESSED_DIR / "topic_timeseries.parquet")
    p.add_argument("--fc-topics", type=Path,
                   default=PROCESSED_DIR / "forecasts.parquet")
    p.add_argument("--ts-trends", type=Path,
                   default=RAW_DIR / "google_trends.parquet")
    p.add_argument("--fc-trends", type=Path,
                   default=PROCESSED_DIR / "trends_forecasts.parquet")
    p.add_argument("--topics-csv", type=Path,
                   default=PROCESSED_DIR / "lda_topics.csv")
    p.add_argument("--out-topics", type=Path,
                   default=METRICS_DIR / "emerging_topics.csv")
    p.add_argument("--out-trends", type=Path,
                   default=METRICS_DIR / "emerging_trends.csv")
    p.add_argument("--as-of", default=None,
                   help="YYYY-MM-DD; backtest mode — ignore data after this date.")
    args = p.parse_args()

    args.out_topics.parent.mkdir(parents=True, exist_ok=True)
    args.out_trends.parent.mkdir(parents=True, exist_ok=True)
    as_of = pd.to_datetime(args.as_of) if args.as_of else None

    df_topics = emerging_topics(
        ts_path=args.ts_topics, fc_path=args.fc_topics,
        topics_path=args.topics_csv, as_of=as_of,
    )
    if not df_topics.empty:
        df_topics.to_csv(args.out_topics, index=False, encoding="utf-8")
        log.info("Saved %s (%d rows)", args.out_topics, len(df_topics))
        log.info("\nTop 5 rising topics:\n%s",
                 df_topics.head(5).to_string(index=False))

    df_trends = emerging_trends(
        ts_path=args.ts_trends, fc_path=args.fc_trends, as_of=as_of,
    )
    if not df_trends.empty:
        df_trends.to_csv(args.out_trends, index=False, encoding="utf-8")
        log.info("Saved %s (%d rows)", args.out_trends, len(df_trends))
        log.info("\nGoogle Trends ranking:\n%s",
                 df_trends.to_string(index=False))


if __name__ == "__main__":
    main()
