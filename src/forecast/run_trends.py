"""Forecast directly on Google Trends weekly series (skips topic modeling).

Each keyword is treated as its own time series. Same model zoo and metrics as
src.forecast.run_all, but driven by the long-format Google Trends parquet.

Usage:
    python -m src.forecast.run_trends \
        --input data/raw/google_trends.parquet \
        --test-size 12
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import METRICS_DIR, PROCESSED_DIR, TEST_SIZE
from src.forecast.run_all import build_model_zoo
from src.forecast.metrics import all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def evaluate_keyword(group: pd.DataFrame, test_size: int) -> tuple[list[dict], pd.DataFrame]:
    g = group.sort_values("date").reset_index(drop=True)
    if len(g) < test_size + 16:
        return [], pd.DataFrame()

    train = g.iloc[:-test_size]
    test = g.iloc[-test_size:]
    df_train = pd.DataFrame({"ds": train["date"], "y": train["interest"]})
    y_test = test["interest"].values

    metrics_rows: list[dict] = []
    forecast_rows: list[dict] = []
    for name, model in build_model_zoo().items():
        try:
            model.fit(df_train)
            pred = model.predict(len(test))
        except Exception as e:  # noqa: BLE001
            log.warning("Model %s failed: %s", name, e)
            continue
        scores = all_metrics(y_test, pred)
        metrics_rows.append({"model": name, **scores})
        for ts, yhat in zip(test["date"].values, pred):
            forecast_rows.append({"model": name, "date": ts, "y_pred": float(yhat)})
    fc = pd.DataFrame(forecast_rows)
    if not fc.empty:
        fc = fc.merge(
            test[["date", "interest"]].rename(columns={"interest": "y_true"}),
            on="date", how="left",
        )
    return metrics_rows, fc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Long-format Google Trends parquet [date, keyword, interest].")
    parser.add_argument("--test-size", type=int, default=TEST_SIZE)
    parser.add_argument("--metrics-out", type=Path,
                        default=METRICS_DIR / "trends_metrics.csv")
    parser.add_argument("--forecasts-out", type=Path,
                        default=PROCESSED_DIR / "trends_forecasts.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df["date"] = pd.to_datetime(df["date"])
    log.info("Loaded %d rows, %d keywords", len(df), df["keyword"].nunique())

    all_m: list[dict] = []
    all_f: list[pd.DataFrame] = []
    for kw, group in tqdm(df.groupby("keyword"), desc="keywords"):
        rows, fc = evaluate_keyword(group, args.test_size)
        for r in rows:
            r["keyword"] = kw
            all_m.append(r)
        if not fc.empty:
            fc["keyword"] = kw
            all_f.append(fc)

    if not all_m:
        log.warning("No metrics produced.")
        return
    metrics_df = pd.DataFrame(all_m)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.metrics_out, index=False)

    summary = (
        metrics_df.groupby("model")[["MAE", "RMSE", "MAPE", "sMAPE"]]
                  .mean().round(3).sort_values("MAE")
    )
    log.info("Average metrics across keywords:\n%s", summary.to_string())
    summary.to_csv(METRICS_DIR / "trends_metrics_summary.csv")

    if all_f:
        fc_df = pd.concat(all_f, ignore_index=True)
        fc_df.to_parquet(args.forecasts_out, index=False)
        log.info("Saved forecasts to %s", args.forecasts_out)


if __name__ == "__main__":
    main()
