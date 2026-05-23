"""Auto-generate case-study figures for the backtest write-up.

Picks a curated set of Google-Trends keywords from the
``detections_with_truth.parquet`` produced by ``src.eval.metrics`` and
draws one PNG per keyword showing:

- Full GT interest history (line plot)
- Vertical marker at the first ``as_of`` where status flipped to Rising
- Shaded "future" window evaluated by the metrics step
- Coloured markers per checkpoint indicating the system's status
  (Rising / Stable / Declining)
- Forecast lines overlaid (best model from each checkpoint)

Usage::

    python -m src.eval.case_studies --max-cases 6
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("case-studies")

BACKTEST_DIR = REPORTS_DIR / "backtest"
CASES_DIR = REPORTS_DIR / "case_studies"

STATUS_COLORS = {"Rising": "#2ca02c", "Stable": "#7f7f7f",
                 "Declining": "#d62728", "Unknown": "#cccccc"}
STATUS_LABELS_UA = {"Rising": "Зростає", "Stable": "Стабільний",
                    "Declining": "Спадає", "Unknown": "Невідомо"}
CASE_KIND_UA = {"success": "успішний", "failure": "помилковий",
                "extra": "додатковий"}


def _pick_cases(det: pd.DataFrame, max_cases: int) -> list[dict]:
    """Return up to ``max_cases`` keywords with a small story.

    Strategy: rank GT keywords by how interesting their evolution is. We
    require at least one Rising detection AND at least one as_of where
    growth was observed. Within those, prefer:

    1. Successes (Rising AND actually_grew)
    2. Big swings (large ``gt_growth`` magnitude)
    3. Mixed (Rising but failed) - 1-2 honest examples
    """
    if det.empty:
        return []

    # Reduce one row per keyword summarising the stats.
    summary = (det.groupby("keyword")
                  .agg(rising_count=("status", lambda s: (s == "Rising").sum()),
                       any_growth=("actually_grew", "max"),
                       max_growth=("gt_growth", "max"),
                       min_growth=("gt_growth", "min"))
                  .reset_index())

    successes = summary[(summary["rising_count"] >= 1) & (summary["any_growth"])]
    successes = successes.sort_values("max_growth", ascending=False)

    failures = summary[(summary["rising_count"] >= 1) & (~summary["any_growth"])]
    failures = failures.sort_values("rising_count", ascending=False)

    cases: list[dict] = []
    for _, row in successes.head(max_cases - 1).iterrows():
        cases.append({"keyword": row["keyword"], "kind": "success"})
    if not failures.empty and len(cases) < max_cases:
        cases.append({"keyword": failures.iloc[0]["keyword"], "kind": "failure"})

    # If we didn't fill all slots, fall back to the largest |growth|.
    used = {c["keyword"] for c in cases}
    if len(cases) < max_cases:
        leftover = summary[~summary["keyword"].isin(used)].copy()
        leftover["abs_growth"] = leftover["max_growth"].abs()
        leftover = leftover.sort_values("abs_growth", ascending=False)
        for _, row in leftover.head(max_cases - len(cases)).iterrows():
            cases.append({"keyword": row["keyword"], "kind": "extra"})
    return cases


def _draw_case(
    keyword: str,
    kind: str,
    gt: pd.DataFrame,
    detections: pd.DataFrame,
    out_dir: Path,
    horizon_weeks: int = 8,
) -> Path | None:
    """Draw one case-study PNG."""
    sub_gt = gt[gt["keyword"] == keyword].sort_values("date")
    sub_det = (detections[detections["keyword"] == keyword]
               .sort_values("as_of"))
    if sub_gt.empty or sub_det.empty:
        return None

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(sub_gt["date"], sub_gt["interest"],
            color="#1f77b4", linewidth=1.6,
            label="Інтерес Google Trends")
    ax.fill_between(sub_gt["date"], 0, sub_gt["interest"],
                    color="#1f77b4", alpha=0.08)

    # Markers per checkpoint.
    for _, r in sub_det.iterrows():
        c = STATUS_COLORS.get(r.get("status", "Unknown"), "#888888")
        ax.axvline(r["as_of"], color=c, alpha=0.35, linewidth=0.8)

    # First Rising marker.
    rising = sub_det[sub_det["status"] == "Rising"]
    if not rising.empty:
        first_rising = rising.iloc[0]
        ax.axvline(first_rising["as_of"], color="#2ca02c",
                   linewidth=2.0, linestyle="--",
                   label=f"Перший сигнал зростання: {first_rising['as_of'].date()}")
        # Highlight evaluation horizon.
        end = first_rising["as_of"] + pd.Timedelta(weeks=horizon_weeks)
        ax.axvspan(first_rising["as_of"], end,
                   color="#2ca02c", alpha=0.10,
                   label=f"Вікно перевірки: +{horizon_weeks} тижнів")

    title = f"Кейс: {keyword} ({CASE_KIND_UA.get(kind, kind)})"
    if "gt_growth" in sub_det.columns and rising is not None and not rising.empty:
        first = rising.iloc[0]
        if pd.notna(first.get("gt_growth")):
            title += f" - спостережене зростання +{first['gt_growth']*100:.0f}% за {horizon_weeks} тижнів"
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Інтерес Google Trends (0-100)")
    ax.grid(True, alpha=0.25)

    # Custom legend (status colours + lines).
    handles, labels = ax.get_legend_handles_labels()
    for status_name, color in STATUS_COLORS.items():
        if (sub_det["status"] == status_name).any():
            from matplotlib.lines import Line2D
            handles.append(Line2D([0], [0], color=color, linewidth=2,
                                  alpha=0.6,
                                  label=STATUS_LABELS_UA.get(status_name, status_name)))
            labels.append(STATUS_LABELS_UA.get(status_name, status_name))
    ax.legend(handles, labels, loc="upper left", fontsize=9, frameon=True)

    fig.tight_layout()
    out_path = out_dir / f"{keyword.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _short(text: str, n: int = 40) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text if len(text) <= n else text[: n - 1] + "..."


def _pick_topic_cases(det: pd.DataFrame, max_cases: int) -> list[dict]:
    """Same logic as _pick_cases but operating on topic_id detections."""
    if det.empty:
        return []
    det = det.copy()
    det["label"] = det["keywords"].fillna("").map(lambda s: _short(s, 50))

    summary = (det.groupby(["topic_id", "label"])
                  .agg(rising_count=("status", lambda s: (s == "Rising").sum()),
                       any_growth=("actually_grew", "max"),
                       max_growth=("ts_growth", "max"),
                       min_growth=("ts_growth", "min"))
                  .reset_index())

    successes = (summary[(summary["rising_count"] >= 1) & (summary["any_growth"])]
                 .sort_values("max_growth", ascending=False))
    failures = (summary[(summary["rising_count"] >= 1) & (~summary["any_growth"])]
                .sort_values("rising_count", ascending=False))

    cases: list[dict] = []
    for _, row in successes.head(max(1, max_cases - 2)).iterrows():
        cases.append({"kind": "success", "topic_id": int(row["topic_id"]),
                      "label": row["label"]})
    if not failures.empty and len(cases) < max_cases:
        r = failures.iloc[0]
        cases.append({"kind": "failure", "topic_id": int(r["topic_id"]),
                      "label": r["label"]})

    used = {c["topic_id"] for c in cases}
    if len(cases) < max_cases:
        leftover = summary[~summary["topic_id"].isin(used)].copy()
        leftover["abs_growth"] = leftover["max_growth"].abs()
        leftover = leftover.sort_values("abs_growth", ascending=False)
        for _, row in leftover.head(max_cases - len(cases)).iterrows():
            cases.append({"kind": "extra", "topic_id": int(row["topic_id"]),
                          "label": row["label"]})
    return cases


def _draw_topic_case(
    case: dict,
    ts: pd.DataFrame,
    detections: pd.DataFrame,
    out_dir: Path,
    horizon_weeks: int = 8,
) -> Path | None:
    tid = case["topic_id"]
    sub_ts = ts[ts["topic_id"] == tid].sort_values("period")
    sub_det = (detections[detections["topic_id"] == tid]
               .sort_values("as_of"))
    if sub_ts.empty or sub_det.empty:
        return None

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(sub_ts["period"], sub_ts["count"],
            color="#1f77b4", linewidth=1.6,
            label="Кількість документів теми за тиждень")
    ax.fill_between(sub_ts["period"], 0, sub_ts["count"],
                    color="#1f77b4", alpha=0.08)

    for _, r in sub_det.iterrows():
        c = STATUS_COLORS.get(r.get("status", "Unknown"), "#888888")
        ax.axvline(r["as_of"], color=c, alpha=0.35, linewidth=0.8)

    rising = sub_det[sub_det["status"] == "Rising"]
    title = f"Тема #{tid} ({CASE_KIND_UA.get(case['kind'], case['kind'])}): {case['label']}"
    if not rising.empty:
        first_rising = rising.iloc[0]
        ax.axvline(first_rising["as_of"], color="#2ca02c",
                   linewidth=2.0, linestyle="--",
                   label=f"Перший сигнал зростання: {first_rising['as_of'].date()}")
        end = first_rising["as_of"] + pd.Timedelta(weeks=horizon_weeks)
        ax.axvspan(first_rising["as_of"], end,
                   color="#2ca02c", alpha=0.10,
                   label=f"Вікно перевірки: +{horizon_weeks} тижнів")
        if pd.notna(first_rising.get("ts_growth")):
            title += f" - спостережена зміна {first_rising['ts_growth']*100:+.0f}% за {horizon_weeks} тижнів"
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Документів на тиждень")
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    for status_name, color in STATUS_COLORS.items():
        if (sub_det["status"] == status_name).any():
            handles.append(Line2D([0], [0], color=color, linewidth=2,
                                  alpha=0.6,
                                  label=STATUS_LABELS_UA.get(status_name, status_name)))
            labels.append(STATUS_LABELS_UA.get(status_name, status_name))
    ax.legend(handles, labels, loc="upper left", fontsize=9, frameon=True)

    fig.tight_layout()
    fname = f"topic_{tid:02d}_{case['label'][:20].strip().replace(' ', '_').replace(',', '')}.png"
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-cases", type=int, default=6)
    p.add_argument("--horizon", type=int, default=8)
    args = p.parse_args()

    CASES_DIR.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict] = []

    # ---- Topic-based cases (primary, since GT has only 3 keywords) ----
    topic_truth = BACKTEST_DIR / "topic_detections_with_truth.parquet"
    if topic_truth.exists():
        det_t = pd.read_parquet(topic_truth)
        det_t["as_of"] = pd.to_datetime(det_t["as_of"])
        ts = pd.read_parquet(PROCESSED_DIR / "topic_timeseries.parquet")
        ts["period"] = pd.to_datetime(ts["period"])
        cases = _pick_topic_cases(det_t, args.max_cases)
        log.info("Drawing %d topic cases", len(cases))
        for c in cases:
            path = _draw_topic_case(c, ts, det_t, CASES_DIR,
                                    horizon_weeks=args.horizon)
            if path is None:
                continue
            sub = det_t[det_t["topic_id"] == c["topic_id"]].sort_values("as_of")
            rising = sub[sub["status"] == "Rising"]
            first = rising.iloc[0] if not rising.empty else None
            index_rows.append({
                "channel": "topic",
                "id": c["topic_id"],
                "label": c["label"],
                "kind": c["kind"],
                "n_detections": int(len(sub)),
                "n_rising": int((sub["status"] == "Rising").sum()),
                "first_rising_as_of": first["as_of"].date() if first is not None else None,
                "first_rising_growth": (round(first["ts_growth"], 3)
                                        if first is not None and pd.notna(first.get("ts_growth"))
                                        else None),
                "max_growth": (round(sub["ts_growth"].max(), 3)
                               if "ts_growth" in sub.columns else None),
                "image": path.name,
            })

    # ---- GT-keyword cases (secondary) ----
    truth_path = BACKTEST_DIR / "detections_with_truth.parquet"
    if truth_path.exists():
        det = pd.read_parquet(truth_path)
        det["as_of"] = pd.to_datetime(det["as_of"])
        gt = pd.read_parquet(RAW_DIR / "google_trends.parquet")
        gt["date"] = pd.to_datetime(gt["date"])
        cases = _pick_cases(det, max_cases=min(3, args.max_cases))
        log.info("Drawing %d GT-keyword cases", len(cases))
        for c in cases:
            path = _draw_case(c["keyword"], c["kind"], gt, det, CASES_DIR,
                              horizon_weeks=args.horizon)
            if path is None:
                continue
            sub = det[det["keyword"] == c["keyword"]].sort_values("as_of")
            rising = sub[sub["status"] == "Rising"]
            first = rising.iloc[0] if not rising.empty else None
            index_rows.append({
                "channel": "trend",
                "id": c["keyword"],
                "label": c["keyword"],
                "kind": c["kind"],
                "n_detections": int(len(sub)),
                "n_rising": int((sub["status"] == "Rising").sum()),
                "first_rising_as_of": first["as_of"].date() if first is not None else None,
                "first_rising_growth": (round(first["gt_growth"], 3)
                                        if first is not None and pd.notna(first.get("gt_growth"))
                                        else None),
                "max_growth": (round(sub["gt_growth"].max(), 3)
                               if "gt_growth" in sub.columns else None),
                "image": path.name,
            })

    if index_rows:
        idx = pd.DataFrame(index_rows)
        idx_path = CASES_DIR / "index.csv"
        idx.to_csv(idx_path, index=False, encoding="utf-8")
        log.info("Saved %s:\n%s", idx_path, idx.to_string(index=False))
    else:
        log.warning("No cases drawn.")


if __name__ == "__main__":
    main()
