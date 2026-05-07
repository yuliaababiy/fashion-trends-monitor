"""TF-IDF spike detection for emerging vocabulary.

Reads ``data/interim/clean.parquet``, builds weekly term-frequency tables
and flags terms whose count in the most recent week is anomalously high
compared with the trailing baseline (mean + std).

Output: ``reports/metrics/spike_terms.csv`` with columns
    term, recent_count, baseline_mean, baseline_std, z_score,
    spike_ratio, weeks_seen, first_seen.

Run::

    python -m src.analysis.spikes
"""
from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import INTERIM_DIR, METRICS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Tunables -------------------------------------------------------------------
RECENT_WEEKS = 1            # how many trailing weeks form the "recent" window
BASELINE_WEEKS = 12         # baseline window length (weeks before recent)
MIN_RECENT_COUNT = 8        # minimum mentions in recent window to consider
MIN_WEEKS_SEEN = 1          # minimum weeks in baseline that the term appeared
Z_THRESHOLD = 3.0           # z-score over baseline mean+std
TOP_N = 50                  # rows to keep in the output CSV
TOKEN_MIN_LEN = 3
# ---------------------------------------------------------------------------

# Generic English / fashion-corpus filler tokens.
STOP_EXTRA = {
    "say", "said", "like", "look", "thing", "year", "day", "time", "week",
    "new", "good", "bad", "people", "make", "made", "use", "used", "know",
    "want", "think", "go", "going", "really", "also", "lot", "many", "much",
    "thing", "way", "even", "still", "back", "first", "last", "next", "old",
    "great", "well", "actually", "find", "found", "post", "comment", "edit",
    "update", "deleted", "removed", "amp", "http", "https",
}

_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_-]{%d,}$" % (TOKEN_MIN_LEN - 1))


def _is_useful(tok: str) -> bool:
    if tok in STOP_EXTRA:
        return False
    return bool(_TOKEN_RE.match(tok))


def _weekly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-form DataFrame: week, term, count."""
    rows: list[dict] = []
    for week, grp in df.groupby("week"):
        c: Counter[str] = Counter()
        for toks in grp["tokens"].dropna():
            for t in toks.split():
                if _is_useful(t):
                    c[t] += 1
        for term, n in c.items():
            rows.append({"week": week, "term": term, "count": n})
    return pd.DataFrame(rows)


def detect_spikes(input_path: Path | None = None,
                  as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    src = Path(input_path) if input_path else INTERIM_DIR / "clean.parquet"
    if not src.exists():
        log.warning("No %s — run preprocessor first", src)
        return pd.DataFrame()

    df = pd.read_parquet(src, columns=["tokens", "week"])
    if as_of is not None:
        before = len(df)
        df = df[pd.to_datetime(df["week"]) <= pd.to_datetime(as_of)].copy()
        log.info("as_of=%s: filtered %d -> %d rows", as_of, before, len(df))
    if df.empty:
        return pd.DataFrame()

    weekly = _weekly_counts(df)
    if weekly.empty:
        return pd.DataFrame()

    weeks = sorted(weekly["week"].unique())
    if len(weeks) < BASELINE_WEEKS + RECENT_WEEKS:
        log.info("Need >= %d weeks of history, have %d — skipping",
                 BASELINE_WEEKS + RECENT_WEEKS, len(weeks))
        return pd.DataFrame()

    recent_weeks = weeks[-RECENT_WEEKS:]
    baseline_weeks = weeks[-(RECENT_WEEKS + BASELINE_WEEKS):-RECENT_WEEKS]

    recent = (weekly[weekly["week"].isin(recent_weeks)]
              .groupby("term")["count"].sum().rename("recent_count"))
    base = weekly[weekly["week"].isin(baseline_weeks)]
    base_stats = (base.groupby("term")["count"]
                  .agg(["mean", "std", "count"])
                  .rename(columns={"mean": "baseline_mean",
                                   "std": "baseline_std",
                                   "count": "weeks_seen"}))
    first_seen = (weekly.groupby("term")["week"].min().rename("first_seen"))

    out = pd.concat([recent, base_stats, first_seen], axis=1)
    out["recent_count"] = out["recent_count"].fillna(0)
    out["baseline_mean"] = out["baseline_mean"].fillna(0)
    out["baseline_std"] = out["baseline_std"].fillna(0)
    out["weeks_seen"] = out["weeks_seen"].fillna(0).astype(int)

    # z-score with epsilon to avoid div by zero
    eps = 1.0
    out["z_score"] = (out["recent_count"] - out["baseline_mean"]) / (
        out["baseline_std"] + eps
    )
    out["spike_ratio"] = out["recent_count"] / (out["baseline_mean"] + eps)

    spikes = out[
        (out["recent_count"] >= MIN_RECENT_COUNT)
        & (out["weeks_seen"] >= MIN_WEEKS_SEEN)
        & (out["z_score"] >= Z_THRESHOLD)
    ].copy()

    spikes = spikes.sort_values("z_score", ascending=False).head(TOP_N)
    spikes = spikes.reset_index().rename(columns={"index": "term"})
    spikes = spikes[[
        "term", "recent_count", "baseline_mean", "baseline_std",
        "z_score", "spike_ratio", "weeks_seen", "first_seen",
    ]].round({"baseline_mean": 2, "baseline_std": 2,
              "z_score": 2, "spike_ratio": 2})
    return spikes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INTERIM_DIR / "clean.parquet")
    parser.add_argument("--output", type=Path, default=METRICS_DIR / "spike_terms.csv")
    parser.add_argument("--as-of", default=None,
                        help="YYYY-MM-DD; only consider data up to this date (backtest mode).")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    as_of = pd.to_datetime(args.as_of) if args.as_of else None
    df = detect_spikes(args.input, as_of=as_of)
    out = args.output
    if df.empty:
        log.info("No spikes detected (or insufficient history).")
        # Still write empty file so consumers see it.
        pd.DataFrame(columns=[
            "term", "recent_count", "baseline_mean", "baseline_std",
            "z_score", "spike_ratio", "weeks_seen", "first_seen",
        ]).to_csv(out, index=False, encoding="utf-8")
        return
    df.to_csv(out, index=False, encoding="utf-8")
    log.info("Saved %s (%d rows)", out, len(df))
    log.info("\nTop 10 spikes:\n%s", df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
