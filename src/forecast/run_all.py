"""Run all forecasting models per topic, compute metrics, save results.

Workflow:
1. Load topic time series (long format: period, topic_id, count, ...).
2. For each topic, split into train/test (last N points).
3. Fit each model on train, forecast `len(test)` periods, compare.
4. Save:
   - reports/metrics/forecast_metrics.csv  (per topic × model)
   - data/processed/forecasts.parquet      (per topic × model × period predictions)

Usage:
    python -m src.forecast.run_all \
        --input data/processed/topic_timeseries.parquet \
        --test-size 12
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import METRICS_DIR, PROCESSED_DIR, TEST_SIZE
from src.forecast.metrics import all_metrics
from src.forecast.models import (
    ARIMAModel,
    LSTMModel,
    MovingAverage,
    NaiveLast,
    ProphetModel,
    XGBLagModel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def build_model_zoo() -> dict:
    return {
        "Naive": NaiveLast(),
        "MA(4)": MovingAverage(window=4),
        "ARIMA(1,1,1)": ARIMAModel(order=(1, 1, 1)),
        "SARIMA(1,1,1)(1,0,1,52)": ARIMAModel(order=(1, 1, 1), seasonal_order=(1, 0, 1, 52)),
        "Prophet": ProphetModel(),
        "XGBoost-lag8": XGBLagModel(lags=8),
        "LSTM": LSTMModel(lags=12, epochs=200),
    }


def evaluate_topic(
    series: pd.DataFrame,
    test_size: int,
    value_col: str = "count",
) -> tuple[list[dict], pd.DataFrame]:
    """Train and evaluate every model on one topic series."""
    series = series.sort_values("period").reset_index(drop=True)
    if len(series) < test_size + 16:
        return [], pd.DataFrame()

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    df_train = pd.DataFrame({"ds": train["period"], "y": train[value_col]})
    y_test = test[value_col].values

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
        for ts, yhat in zip(test["period"].values, pred):
            forecast_rows.append({
                "model": name, "period": ts, "y_pred": float(yhat),
            })
    forecasts_df = pd.DataFrame(forecast_rows)
    if not forecasts_df.empty:
        forecasts_df = forecasts_df.merge(
            test[["period", value_col]].rename(columns={value_col: "y_true"}),
            on="period", how="left",
        )
    return metrics_rows, forecasts_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=PROCESSED_DIR / "topic_timeseries.parquet")
    parser.add_argument("--test-size", type=int, default=TEST_SIZE)
    parser.add_argument("--value-col", default="count")
    parser.add_argument("--metrics-out", type=Path,
                        default=METRICS_DIR / "forecast_metrics.csv")
    parser.add_argument("--forecasts-out", type=Path,
                        default=PROCESSED_DIR / "forecasts.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    log.info("Loaded %d rows, %d topics", len(df), df["topic_id"].nunique())

    all_metrics_rows: list[dict] = []
    all_forecasts: list[pd.DataFrame] = []

    for topic_id, group in tqdm(df.groupby("topic_id"), desc="topics"):
        rows, fc = evaluate_topic(group, args.test_size, args.value_col)
        for r in rows:
            r["topic_id"] = int(topic_id)
            all_metrics_rows.append(r)
        if not fc.empty:
            fc["topic_id"] = int(topic_id)
            all_forecasts.append(fc)

    metrics_df = pd.DataFrame(all_metrics_rows)
    if metrics_df.empty:
        log.warning("No metrics produced (series too short?).")
        return

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.metrics_out, index=False)
    log.info("Saved per-topic metrics to %s", args.metrics_out)

    summary = (
        metrics_df.groupby("model")[["MAE", "RMSE", "MAPE", "sMAPE"]]
                  .mean().round(3).sort_values("MAE")
    )
    log.info("Average metrics across topics:\n%s", summary.to_string())
    summary.to_csv(METRICS_DIR / "forecast_metrics_summary.csv")

    if all_forecasts:
        fc_df = pd.concat(all_forecasts, ignore_index=True)
        fc_df.to_parquet(args.forecasts_out, index=False)
        log.info("Saved forecasts to %s", args.forecasts_out)


if __name__ == "__main__":
    main()
