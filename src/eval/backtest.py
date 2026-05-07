"""Run backtest harness across N retrospective `as_of` checkpoints.

For each checkpoint (e.g. first day of each month over the last year), this
script:

1. Trains forecast models on data up to ``as_of`` (rolling origin) and
   evaluates them on the next ``--horizon`` weeks.
2. Recomputes emerging-topic / emerging-trend rankings as if "today" were
   ``as_of``, using the rolling-origin forecasts above.
3. Recomputes TF-IDF spike detections at ``as_of``.

All artifacts are written under ``reports/backtest/<YYYY-MM-DD>/``. The
production state under ``reports/`` and ``data/processed/`` is **not**
touched, so live monitoring keeps working in parallel.

Usage::

    python -m src.eval.backtest \
        --start 2025-05-01 --end 2026-04-01 \
        --step monthly --horizon 8

A final ``reports/backtest/all_detections.parquet`` aggregates every
detection across checkpoints with an ``as_of`` column, ready for
metric-computation in ``src.eval.metrics``.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backtest")

PY = sys.executable
ROOT = Path(__file__).resolve().parents[2]
BACKTEST_DIR = REPORTS_DIR / "backtest"


def _checkpoints(start: str, end: str, step: str) -> list[pd.Timestamp]:
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if step == "monthly":
        return list(pd.date_range(start_ts, end_ts, freq="MS"))
    if step == "weekly":
        return list(pd.date_range(start_ts, end_ts, freq="W-MON"))
    raise ValueError(f"Unknown step: {step}")


def _run(cmd: list[str], optional: bool = False) -> bool:
    log.info("$ %s", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return True
    except subprocess.CalledProcessError as e:
        log.warning("Command failed (optional=%s): %s", optional, e)
        if not optional:
            raise
        return False


def run_one(as_of: pd.Timestamp, horizon: int) -> Path:
    """Run the full backtest pipeline for a single checkpoint."""
    tag = as_of.strftime("%Y-%m-%d")
    out_dir = BACKTEST_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    fc_topics = out_dir / "forecasts.parquet"
    fc_trends = out_dir / "trends_forecasts.parquet"
    metrics_topics = out_dir / "forecast_metrics.csv"
    metrics_trends = out_dir / "trends_metrics.csv"
    em_topics = out_dir / "emerging_topics.csv"
    em_trends = out_dir / "emerging_trends.csv"
    spikes = out_dir / "spike_terms.csv"

    log.info("=== %s ===", tag)

    # 1) Rolling-origin forecasts on topic timeseries.
    _run([
        PY, "-m", "src.forecast.run_all",
        "--input", str(PROCESSED_DIR / "topic_timeseries.parquet"),
        "--as-of", tag,
        "--horizon", str(horizon),
        "--metrics-out", str(metrics_topics),
        "--forecasts-out", str(fc_topics),
    ], optional=True)

    # 2) Rolling-origin forecasts on Google Trends keywords.
    _run([
        PY, "-m", "src.forecast.run_trends",
        "--input", str(RAW_DIR / "google_trends.parquet"),
        "--as-of", tag,
        "--horizon", str(horizon),
        "--metrics-out", str(metrics_trends),
        "--forecasts-out", str(fc_trends),
    ], optional=True)

    # 3) Emerging analysis (topics + trends) at as_of.
    cmd = [
        PY, "-m", "src.analysis.emerging",
        "--as-of", tag,
        "--out-topics", str(em_topics),
        "--out-trends", str(em_trends),
    ]
    if fc_topics.exists():
        cmd += ["--fc-topics", str(fc_topics)]
    if fc_trends.exists():
        cmd += ["--fc-trends", str(fc_trends)]
    _run(cmd, optional=True)

    # 4) Spike detection at as_of.
    _run([
        PY, "-m", "src.analysis.spikes",
        "--as-of", tag,
        "--output", str(spikes),
    ], optional=True)

    return out_dir


def aggregate() -> Path:
    """Concat detections from all checkpoint directories into a single parquet."""
    rows_topics: list[pd.DataFrame] = []
    rows_trends: list[pd.DataFrame] = []
    rows_spikes: list[pd.DataFrame] = []
    for sub in sorted(BACKTEST_DIR.iterdir()):
        if not sub.is_dir():
            continue
        tag = sub.name
        try:
            pd.to_datetime(tag)
        except (ValueError, TypeError):
            continue
        for fname, store in (
            ("emerging_topics.csv", rows_topics),
            ("emerging_trends.csv", rows_trends),
            ("spike_terms.csv", rows_spikes),
        ):
            p = sub / fname
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
            except Exception as e:  # noqa: BLE001
                log.warning("Could not read %s: %s", p, e)
                continue
            if df.empty:
                continue
            df["as_of"] = pd.to_datetime(tag)
            store.append(df)

    out = BACKTEST_DIR / "all_detections.parquet"
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {}
    if rows_topics:
        bundle["emerging_topics"] = pd.concat(rows_topics, ignore_index=True)
    if rows_trends:
        bundle["emerging_trends"] = pd.concat(rows_trends, ignore_index=True)
    if rows_spikes:
        bundle["spike_terms"] = pd.concat(rows_spikes, ignore_index=True)

    # Save each table side-by-side as separate files for simpler downstream use.
    for name, df in bundle.items():
        path = BACKTEST_DIR / f"all_{name}.parquet"
        df.to_parquet(path, index=False)
        log.info("Aggregated %s: %d rows -> %s", name, len(df), path)

    # Also dump a unified slim view for quick eyeballing.
    if "emerging_topics" in bundle:
        bundle["emerging_topics"].to_parquet(out, index=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-05-01")
    p.add_argument("--end", default="2026-04-01")
    p.add_argument("--step", default="monthly", choices=["monthly", "weekly"])
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--only", default=None,
                   help="Comma-separated list of YYYY-MM-DD dates; overrides start/end/step.")
    p.add_argument("--no-aggregate", action="store_true",
                   help="Do not run the final aggregation step.")
    args = p.parse_args()

    if args.only:
        checkpoints = [pd.to_datetime(x.strip()) for x in args.only.split(",") if x.strip()]
    else:
        checkpoints = _checkpoints(args.start, args.end, args.step)

    log.info("Running backtest across %d checkpoints (horizon=%dw): %s",
             len(checkpoints), args.horizon,
             [c.strftime("%Y-%m-%d") for c in checkpoints])

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    for c in checkpoints:
        try:
            run_one(c, horizon=args.horizon)
        except Exception as e:  # noqa: BLE001
            log.exception("Checkpoint %s failed: %s", c.strftime("%Y-%m-%d"), e)

    if not args.no_aggregate:
        aggregate()


if __name__ == "__main__":
    main()
