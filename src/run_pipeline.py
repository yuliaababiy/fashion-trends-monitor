"""End-to-end pipeline runner.

Runs every step of the project sequentially, logging progress and
gracefully skipping optional stages on failure.

Usage:
    python -m src.run_pipeline
    python -m src.run_pipeline --skip-collect      # reuse existing parquet files
    python -m src.run_pipeline --skip-trends       # skip Google Trends only
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("pipeline")

PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]


def step(title: str, cmd: list[str], optional: bool = False) -> bool:
    log.info("=" * 70)
    log.info("STEP: %s", title)
    log.info("CMD : %s", " ".join(cmd))
    log.info("=" * 70)
    t0 = time.time()
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
        log.info("OK in %.1fs: %s", time.time() - t0, title)
        return True
    except subprocess.CalledProcessError as e:
        if optional:
            log.warning("Optional step failed (%s): %s — continuing", title, e)
            return False
        log.error("Step failed: %s", title)
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-collect", action="store_true",
                    help="Reuse existing data/raw/*.parquet")
    ap.add_argument("--skip-trends", action="store_true",
                    help="Skip Google Trends collection and forecasting")
    ap.add_argument("--guardian-from", default="2021-01-01")
    ap.add_argument("--guardian-pages", type=int, default=60)
    ap.add_argument("--trends-timeframe", default="today 5-y")
    args = ap.parse_args()

    # ---- collection ------------------------------------------------------
    if not args.skip_collect:
        step(
            "1/9 Guardian articles",
            [PY, "-m", "src.collect.guardian",
             "--from-date", args.guardian_from,
             "--pages", str(args.guardian_pages)],
        )
        step(
            "2/9 News API articles (last 30 days)",
            [PY, "-m", "src.collect.newsapi"],
            optional=True,
        )
        step(
            "2b/9 Reddit (public JSON, no auth)",
            [PY, "-m", "src.collect.reddit_json",
             "--listing", "new", "--pages", "5"],
            optional=True,
        )
        step(
            "2c/9 Mastodon hashtags",
            [PY, "-m", "src.collect.mastodon", "--pages", "3"],
            optional=True,
        )
        if not args.skip_trends:
            step(
                "3/9 Google Trends",
                [PY, "-m", "src.collect.google_trends",
                 "--timeframe", args.trends_timeframe],
                optional=True,
            )
    else:
        log.info("Skipping collection.")

    # ---- combine raw post-like sources -----------------------------------
    inputs = []
    for f in ["guardian.parquet", "newsapi.parquet",
              "reddit.parquet", "mastodon.parquet"]:
        p = ROOT / "data" / "raw" / f
        if p.exists():
            inputs.append(str(p))
    if not inputs:
        log.error("No input parquet found in data/raw/. Aborting.")
        sys.exit(1)
    step(
        "4/9 Combine sources",
        [PY, "-m", "src.collect.combine", "--inputs", *inputs],
    )

    # ---- preprocessing ---------------------------------------------------
    step("5/9 Preprocess (clean + lemmatize)", [PY, "-m", "src.preprocess.cleaner"])

    # ---- topic modeling --------------------------------------------------
    step("6/9 LDA topic modeling", [PY, "-m", "src.topics.run_lda"])

    # ---- timeseries ------------------------------------------------------
    step(
        "7/9 Build weekly topic timeseries",
        [PY, "-m", "src.topics.build_timeseries",
         "--input", "data/processed/lda_doc_topics.parquet"],
    )

    # ---- forecasting on topics ------------------------------------------
    step("8/9 Forecast topic timeseries (all models)",
         [PY, "-m", "src.forecast.run_all"])

    # ---- forecasting on Google Trends ------------------------------------
    trends_path = ROOT / "data" / "raw" / "google_trends.parquet"
    if trends_path.exists() and not args.skip_trends:
        step(
            "9/9 Forecast Google Trends series",
            [PY, "-m", "src.forecast.run_trends", "--input", str(trends_path)],
            optional=True,
        )
    else:
        log.info("9/9 Skipping Google Trends forecasting (no data).")

    # ---- emerging trend analysis -----------------------------------------
    step("10/10 Emerging trend ranking",
         [PY, "-m", "src.analysis.emerging"], optional=True)

    log.info("=" * 70)
    log.info("PIPELINE COMPLETE.")
    log.info("Run dashboard: streamlit run app.py")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
