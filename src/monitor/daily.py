"""Daily monitoring pipeline.

Pulls a small batch of fresh data from each source incrementally,
re-runs cleaning + LDA-transform + timeseries + forecasts + emerging
analysis, then diffs the new ``emerging_topics.csv`` against the
previously saved snapshot. Any new topics that became *Rising* (or
already-Rising topics whose forecast jumped) are pushed to the
configured Telegram chats.

Usage::

    python -m src.monitor.daily
    python -m src.monitor.daily --skip-collect       # just rerun analytics
    python -m src.monitor.daily --no-alerts          # do not call Telegram

Designed to be invoked from a GitHub Actions cron workflow.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import METRICS_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("monitor")

PY = sys.executable
ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = REPORTS_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
PREVIOUS_STATE = STATE_DIR / "emerging_yesterday.csv"

# Forecast-pct delta (in percentage points) considered noteworthy enough
# to send an alert when no Status change happened.
NOTEWORTHY_DELTA = 10.0


# ---------------------------------------------------------------------------
def _step(title: str, cmd: list[str], optional: bool = False) -> bool:
    log.info("STEP: %s", title)
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return True
    except subprocess.CalledProcessError as e:
        if optional:
            log.warning("Optional step failed: %s (%s)", title, e)
            return False
        raise


def collect_incremental(skip_newsapi: bool = False) -> None:
    """Run all source collectors in incremental mode."""
    if os.getenv("GUARDIAN_API_KEY"):
        _step(
            "Guardian (incremental)",
            [PY, "-m", "src.collect.guardian",
             "--incremental", "--pages", "5"],
            optional=True,
        )
    else:
        log.info("GUARDIAN_API_KEY missing; skipping Guardian.")

    if os.getenv("NEWSAPI_KEY") and not skip_newsapi:
        _step(
            "NewsAPI (incremental)",
            [PY, "-m", "src.collect.newsapi", "--incremental"],
            optional=True,
        )
    else:
        log.info("NEWSAPI_KEY missing or skipped; skipping NewsAPI.")

    # Reddit: prefer OAuth (works on data-center IPs); fall back to the
    # public JSON endpoint when running locally without OAuth credentials.
    has_oauth = all(os.getenv(k) for k in (
        "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET",
        "REDDIT_USERNAME", "REDDIT_PASSWORD", "REDDIT_USER_AGENT",
    ))
    if has_oauth:
        _step(
            "Reddit OAuth (incremental)",
            [PY, "-m", "src.collect.reddit_oauth",
             "--listing", "new", "--pages", "3", "--incremental"],
            optional=True,
        )
    elif os.getenv("GITHUB_ACTIONS") == "true":
        log.info("No Reddit OAuth credentials and running on GitHub "
                 "Actions — skipping Reddit (public JSON is blocked on "
                 "data-center IPs).")
    else:
        _step(
            "Reddit JSON (incremental)",
            [PY, "-m", "src.collect.reddit_json",
             "--listing", "new", "--pages", "3", "--incremental"],
            optional=True,
        )

    _step(
        "Mastodon (incremental)",
        [PY, "-m", "src.collect.mastodon",
         "--pages", "2", "--incremental"],
        optional=True,
    )


def find_lda_model() -> Path | None:
    """Locate the most recent saved LDA bundle."""
    candidates = sorted(MODELS_DIR.glob("lda_k*.pkl"))
    if not candidates:
        return None
    return candidates[-1]


def run_analytics(transform_only: bool = True) -> None:
    """Combine sources -> clean -> LDA (transform) -> timeseries -> forecasts -> emerging."""
    inputs: list[str] = []
    for f in ("guardian.parquet", "newsapi.parquet",
              "reddit.parquet", "mastodon.parquet"):
        p = ROOT / "data" / "raw" / f
        if p.exists():
            inputs.append(str(p))
    if not inputs:
        raise RuntimeError("No raw parquet files found.")

    _step("Combine sources",
          [PY, "-m", "src.collect.combine", "--inputs", *inputs])

    _step("Preprocess",
          [PY, "-m", "src.preprocess.cleaner"])

    if transform_only:
        model = find_lda_model()
        if model is None:
            log.warning("No saved LDA model — falling back to full training.")
            _step("LDA (full retrain fallback)",
                  [PY, "-m", "src.topics.run_lda"])
        else:
            _step(f"LDA transform with {model.name}",
                  [PY, "-m", "src.topics.run_lda",
                   "--transform-only", str(model)])
    else:
        _step("LDA (full retrain)",
              [PY, "-m", "src.topics.run_lda"])

    _step("Build topic timeseries",
          [PY, "-m", "src.topics.build_timeseries",
           "--input", "data/processed/lda_doc_topics.parquet"])

    _step("Forecast topics", [PY, "-m", "src.forecast.run_all"])
    _step("Emerging analysis", [PY, "-m", "src.analysis.emerging"],
          optional=True)


# ---------------------------------------------------------------------------
def diff_emerging() -> dict:
    """Return a dict describing how today's emerging table differs from
    the previously saved snapshot. Keys::

        new_rising:  list of dicts (topic_id, keywords, forecast_pct, momentum_pct, status)
        big_changes: list of dicts where forecast_pct moved by NOTEWORTHY_DELTA pp
        all_today:   today's full emerging_topics dataframe
    """
    today_path = METRICS_DIR / "emerging_topics.csv"
    if not today_path.exists():
        log.warning("emerging_topics.csv missing — nothing to diff.")
        return {"new_rising": [], "big_changes": [], "all_today": pd.DataFrame()}

    today = pd.read_csv(today_path)

    if not PREVIOUS_STATE.exists():
        log.info("No previous state; treating all current Rising as new.")
        new_rising = today[today["status"] == "Rising"].to_dict("records")
        return {"new_rising": new_rising, "big_changes": [], "all_today": today}

    prev = pd.read_csv(PREVIOUS_STATE)
    merged = today.merge(
        prev[["topic_id", "status", "forecast_pct", "momentum_pct"]]
            .rename(columns={"status": "status_prev",
                             "forecast_pct": "forecast_pct_prev",
                             "momentum_pct": "momentum_pct_prev"}),
        on="topic_id", how="left",
    )

    new_rising = merged[
        (merged["status"] == "Rising") &
        (merged["status_prev"].fillna("") != "Rising")
    ].to_dict("records")

    big_changes = []
    for _, row in merged.iterrows():
        if pd.isna(row.get("forecast_pct")) or pd.isna(row.get("forecast_pct_prev")):
            continue
        delta = row["forecast_pct"] - row["forecast_pct_prev"]
        if abs(delta) >= NOTEWORTHY_DELTA and row["status"] == "Rising":
            big_changes.append({**row.to_dict(), "delta_pp": round(delta, 1)})

    return {"new_rising": new_rising,
            "big_changes": big_changes,
            "all_today": today}


def save_today_as_state() -> None:
    today_path = METRICS_DIR / "emerging_topics.csv"
    if today_path.exists():
        df = pd.read_csv(today_path)
        df.to_csv(PREVIOUS_STATE, index=False)
        log.info("Saved snapshot to %s", PREVIOUS_STATE)


# ---------------------------------------------------------------------------
def format_alert(diff: dict) -> str | None:
    """Build a Markdown message for Telegram, or None if nothing to report."""
    new_rising = diff.get("new_rising", [])
    big_changes = diff.get("big_changes", [])

    if not new_rising and not big_changes:
        return None

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"📈 *Trend monitor — {today}*", ""]

    if new_rising:
        lines.append(f"🚀 *New Rising topics ({len(new_rising)})*:")
        for r in new_rising[:5]:
            kw = (r.get("keywords") or "")[:60]
            fp = r.get("forecast_pct")
            mp = r.get("momentum_pct")
            lines.append(
                f"  • #{int(r['topic_id'])} _{kw}_  "
                f"momentum {mp:+.0f}% / forecast {fp:+.0f}%"
                if fp is not None and not pd.isna(fp)
                else f"  • #{int(r['topic_id'])} _{kw}_  momentum {mp:+.0f}%"
            )
        lines.append("")

    if big_changes:
        lines.append(f"⚡ *Forecast shifts (≥{NOTEWORTHY_DELTA:.0f} pp)*:")
        for r in big_changes[:5]:
            kw = (r.get("keywords") or "")[:60]
            lines.append(
                f"  • #{int(r['topic_id'])} _{kw}_  Δ {r['delta_pp']:+.1f} pp"
            )
        lines.append("")

    lines.append("Run `/trends` in the bot for the full ranking.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-collect", action="store_true",
                   help="Reuse existing data/raw/*.parquet")
    p.add_argument("--no-alerts", action="store_true",
                   help="Do not send Telegram messages")
    p.add_argument("--retrain-lda", action="store_true",
                   help="Re-fit LDA from scratch (use only on weekly).")
    args = p.parse_args()

    if not args.skip_collect:
        collect_incremental()

    run_analytics(transform_only=not args.retrain_lda)

    diff = diff_emerging()
    msg = format_alert(diff)

    if msg:
        log.info("Alert message:\n%s", msg)
        if not args.no_alerts:
            try:
                from src.alerts.telegram import send_alert_to_all
                send_alert_to_all(msg)
            except Exception as e:
                log.error("Failed to send Telegram alert: %s", e)
    else:
        log.info("No alert-worthy changes.")

    save_today_as_state()
    log.info("Daily monitor complete.")


if __name__ == "__main__":
    main()
