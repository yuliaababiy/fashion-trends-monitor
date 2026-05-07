"""Streamlit dashboard for fashion trend prediction.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import METRICS_DIR, PROCESSED_DIR, RAW_DIR, REPORTS_DIR

BACKTEST_DIR = REPORTS_DIR / "backtest"
CASES_DIR = REPORTS_DIR / "case_studies"

st.set_page_config(
    page_title="Прогнозування модних трендів",
    page_icon=":dress:",
    layout="wide",
)

st.title("👗 Прогнозування модних трендів у соціальних мережах")
st.caption(
    "Курсова робота — тематичне моделювання та прогнозування часових рядів на основі "
    "The Guardian, News API та Google Trends."
)

PALETTE = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_topics() -> pd.DataFrame:
    for p in [PROCESSED_DIR / "bertopic_topics.csv",
              PROCESSED_DIR / "lda_topics.csv"]:
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(ttl=600)
def load_topic_ts() -> pd.DataFrame:
    p = PROCESSED_DIR / "topic_timeseries.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["period"] = pd.to_datetime(df["period"])
    return df


@st.cache_data(ttl=600)
def load_topic_forecasts() -> pd.DataFrame:
    p = PROCESSED_DIR / "forecasts.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["period"] = pd.to_datetime(df["period"])
    return df


@st.cache_data(ttl=600)
def load_topic_metrics() -> pd.DataFrame:
    p = METRICS_DIR / "forecast_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=600)
def load_trends_ts() -> pd.DataFrame:
    p = RAW_DIR / "google_trends.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=600)
def load_trends_forecasts() -> pd.DataFrame:
    p = PROCESSED_DIR / "trends_forecasts.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=600)
def load_trends_metrics() -> pd.DataFrame:
    p = METRICS_DIR / "trends_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=600)
def load_emerging_topics() -> pd.DataFrame:
    p = METRICS_DIR / "emerging_topics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=600)
def load_emerging_trends() -> pd.DataFrame:
    p = METRICS_DIR / "emerging_trends.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=600)
def load_spike_terms() -> pd.DataFrame:
    p = METRICS_DIR / "spike_terms.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=600)
def load_backtest_topics() -> pd.DataFrame:
    p = BACKTEST_DIR / "topic_detections_with_truth.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["as_of"] = pd.to_datetime(df["as_of"])
    return df


@st.cache_data(ttl=600)
def load_backtest_trends() -> pd.DataFrame:
    p = BACKTEST_DIR / "detections_with_truth.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["as_of"] = pd.to_datetime(df["as_of"])
    return df


@st.cache_data(ttl=600)
def load_backtest_summary() -> pd.DataFrame:
    p = BACKTEST_DIR / "summary.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=600)
def load_backtest_pak() -> pd.DataFrame:
    p = BACKTEST_DIR / "precision_at_k.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["as_of"] = pd.to_datetime(df["as_of"])
    return df


@st.cache_data(ttl=600)
def load_backtest_models() -> pd.DataFrame:
    p = BACKTEST_DIR / "forecast_models_comparison.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=600)
def load_lead_time_summary() -> pd.DataFrame:
    p = BACKTEST_DIR / "lead_time_summary.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=600)
def load_lead_time_hist() -> pd.DataFrame:
    p = BACKTEST_DIR / "lead_time_hist_topics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=600)
def load_baseline_compare() -> pd.DataFrame:
    p = BACKTEST_DIR / "precision_vs_baseline.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=600)
def load_case_index() -> pd.DataFrame:
    p = CASES_DIR / "index.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def topic_label(tid: int, topics_df: pd.DataFrame) -> str:
    if topics_df.empty:
        return f"Тема {tid}"
    if "Topic" in topics_df.columns:  # BERTopic
        row = topics_df[topics_df["Topic"] == tid]
        if not row.empty:
            return f"{tid} - {row.iloc[0].get('Name', '')}"
    if "topic_id" in topics_df.columns:  # LDA
        row = topics_df[topics_df["topic_id"] == tid]
        if not row.empty:
            kw = str(row.iloc[0].get("keywords", ""))[:60]
            return f"{tid} - {kw}"
    return f"Тема {tid}"


def add_forecast_traces(fig, fc_df, x_col, y_col, models):
    for i, (model, grp) in enumerate(fc_df.groupby("model")):
        if model not in models:
            continue
        fig.add_trace(go.Scatter(
            x=grp[x_col], y=grp[y_col],
            mode="lines+markers", name=f"{model} (прогноз)",
            line=dict(color=PALETTE[i % len(PALETTE)], dash="dash"),
        ))


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_emerging, tab_spikes, tab_topics, tab_trends, tab_compare, tab_backtest = st.tabs(
    ["📈 Тренди, що зростають", "🔥 Сплески слів",
     "🔍 Теми LDA", "🌐 Google Trends", "⚖️ Порівняння моделей",
     "📊 Ретроспектива"]
)


# ============================ EMERGING TAB =================================
def _status_color(s: str) -> str:
    return {"Rising": "#2ca02c", "Declining": "#d62728",
            "Stable": "#7f7f7f"}.get(s, "#7f7f7f")


STATUS_UA = {"Rising": "зростає", "Declining": "спадає", "Stable": "стабільний"}


with tab_emerging:
    st.markdown(
        "**Що зростає, а що — спадає?** Поєднує історичний моментум "
        "(останні 8 тижнів vs попередні 26) з прогнозом найкращої моделі "
        "на наступні 8 тижнів."
    )

    em_topics = load_emerging_topics()
    em_trends = load_emerging_trends()

    st.subheader("Ключові слова Google Trends")
    if em_trends.empty:
        st.info("Спочатку виконайте `python -m src.analysis.emerging`.")
    else:
        cols = st.columns(len(em_trends))
        for col, (_, r) in zip(cols, em_trends.iterrows()):
            with col:
                col.markdown(f"**{r['keyword']}**")
                col.metric("Моментум (останні vs попередні)",
                           f"{r['momentum_pct']:+.1f}%")
                if pd.notna(r["forecast_pct"]):
                    col.metric("Прогноз на 8 тижнів",
                               f"{r['forecast_pct']:+.1f}%")
                col.markdown(
                    f"<span style='background:{_status_color(r['status'])};"
                    f"color:white;padding:4px 10px;border-radius:6px;'>"
                    f"{STATUS_UA.get(r['status'], r['status'])}</span>",
                    unsafe_allow_html=True,
                )

        fig = go.Figure()
        em_sorted = em_trends.sort_values("momentum_pct")
        fig.add_trace(go.Bar(
            x=em_sorted["momentum_pct"], y=em_sorted["keyword"],
            orientation="h", name="Історичний моментум",
            marker_color=[_status_color(s) for s in em_sorted["status"]],
            text=[f"{v:+.0f}%" for v in em_sorted["momentum_pct"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=300, xaxis_title="Моментум, %",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

        st.dataframe(em_trends, width="stretch")

    st.subheader("Теми LDA, впорядковані за прогнозом зростання")
    if em_topics.empty:
        st.info("Спочатку виконайте `python -m src.analysis.emerging`.")
    else:
        rising = em_topics[em_topics["status"] == "Rising"]
        declining = em_topics[em_topics["status"] == "Declining"]
        stable = em_topics[em_topics["status"] == "Stable"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Тем, що зростають", len(rising))
        c2.metric("Стабільних тем", len(stable))
        c3.metric("Тем, що спадають", len(declining))

        st.markdown("##### Топ-10 за історичним моментумом")
        em_named = em_topics.dropna(subset=["keywords"])
        em_named = em_named[em_named["keywords"].astype(str).str.strip() != ""]
        top = em_named.sort_values("momentum_pct", ascending=False).head(10)
        def _short(k: str, n: int = 55) -> str:
            k = str(k)
            return k if len(k) <= n else k[: n - 1].rstrip(", ") + "…"
        labels = [f"#{int(t)} — {_short(k)}" for t, k in
                  zip(top["topic_id"], top["keywords"])]
        fig = go.Figure(go.Bar(
            x=top["momentum_pct"],
            y=labels,
            orientation="h",
            marker_color=[_status_color(s) for s in top["status"]],
            text=[f"{v:+.0f}%" for v in top["momentum_pct"]],
            textposition="outside",
            hovertext=[str(k) for k in top["keywords"]],
            hoverinfo="text+x",
        ))
        fig.update_layout(
            height=560, xaxis_title="Моментум, %",
            yaxis=dict(autorange="reversed", automargin=True, tickfont=dict(size=12)),
            margin=dict(l=10, r=60, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("##### Повний рейтинг")
        em_display = em_topics.copy()
        em_display["keywords"] = em_display["keywords"].fillna("(без ключових слів)")
        st.dataframe(
            em_display,
            width="stretch",
            column_config={
                "keywords": st.column_config.TextColumn(
                    "keywords", width="large"),
            },
        )

        st.caption(
            "**Як читати:** *momentum_pct* — що привертає увагу САМЕ ЗАРАЗ "
            "(останні 8 тижнів vs попередні 26). "
            "*forecast_pct* — прогноз моделі на наступні 8 тижнів "
            "(додатнє значення — очікується подальше зростання). "
            "Теми з високим моментумом, але від'ємним прогнозом — ймовірно близько до піку."
        )


# ============================ SPIKES TAB ===================================
with tab_spikes:
    st.markdown(
        "**Сплески окремих слів (TF-IDF).** Шукаємо слова, частота яких "
        "за останній тиждень аномально вища за базу попередніх 12 тижнів. "
        "Це раннє попередження — слова можуть з'явитися ще до того, як "
        "сформують повноцінну тему LDA."
    )

    spikes = load_spike_terms()
    if spikes.empty:
        st.info(
            "Поки немає `spike_terms.csv`. Запустіть "
            "`python -m src.analysis.spikes` або зачекайте на щоденний пайплайн."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Виявлено сплесків", len(spikes))
        c2.metric("Найвищий z-score", f"{spikes['z_score'].max():.1f}")
        c3.metric("Найбільший ×ratio", f"{spikes['spike_ratio'].max():.1f}")

        st.markdown("##### Топ-15 за z-score")
        top = spikes.sort_values("z_score", ascending=False).head(15)
        fig = go.Figure(go.Bar(
            x=top["z_score"],
            y=top["term"],
            orientation="h",
            marker_color="#d62728",
            text=[f"×{r:.1f}" for r in top["spike_ratio"]],
            textposition="outside",
            hovertext=[
                f"{t}: recent={int(rc)}, base={bm:.1f}±{bs:.1f}"
                for t, rc, bm, bs in zip(
                    top["term"], top["recent_count"],
                    top["baseline_mean"], top["baseline_std"])
            ],
            hoverinfo="text+x",
        ))
        fig.update_layout(
            height=520, xaxis_title="z-score",
            yaxis=dict(autorange="reversed", automargin=True),
            margin=dict(l=10, r=40, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("##### Повна таблиця")
        st.dataframe(spikes, width="stretch")
        st.caption(
            "**z-score** — наскільки останній тиждень відхиляється від "
            "середнього бази (≥3 = аномалія). **spike_ratio** — у скільки "
            "разів частіше слово згадувалось зараз порівняно з базою. "
            "**recent_count** — згадок за останній тиждень."
        )


# ============================ TOPICS TAB ===================================
with tab_topics:
    topics_df = load_topics()
    ts_df = load_topic_ts()
    fc_df = load_topic_forecasts()
    metrics_df = load_topic_metrics()

    if ts_df.empty:
        st.warning("Немає часових рядів тем. Запустіть `python -m src.run_pipeline`.")
    else:
        col_a, col_b = st.columns([2, 3])
        with col_a:
            tids = sorted(ts_df["topic_id"].unique().tolist())
            tid = st.selectbox("Тема", tids,
                               format_func=lambda t: topic_label(t, topics_df))
        with col_b:
            available_models = sorted(fc_df["model"].unique()) if not fc_df.empty else []
            show = st.multiselect("Моделі для відображення", available_models,
                                  default=available_models)

        topic_ts = ts_df[ts_df["topic_id"] == tid].sort_values("period")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Всього згадок", int(topic_ts["count"].sum()))
        c2.metric("Пік за тиждень", int(topic_ts["count"].max()))
        c3.metric("Останній тиждень",
                  int(topic_ts["count"].iloc[-1]) if len(topic_ts) else 0)
        recent = topic_ts.tail(8)["count"].mean() if len(topic_ts) >= 8 else 0
        older = topic_ts.iloc[-16:-8]["count"].mean() if len(topic_ts) >= 16 else 0
        trend_pct = ((recent - older) / older * 100.0) if older > 0 else 0.0
        c4.metric("Тренд за 8 тижнів", f"{trend_pct:+.1f}%")

        st.subheader(f"Часовий ряд — {topic_label(tid, topics_df)}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=topic_ts["period"], y=topic_ts["count"],
            mode="lines", name="Історія",
            line=dict(color="#1f77b4", width=2),
        ))
        if not fc_df.empty:
            sub = fc_df[fc_df["topic_id"] == tid]
            add_forecast_traces(fig, sub, "period", "y_pred", show)
        fig.update_layout(height=450, xaxis_title="Дата",
                          yaxis_title="Публікацій на тиждень",
                          legend=dict(orientation="h", y=-0.2),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        if not metrics_df.empty:
            st.subheader("Точність прогнозу (для цієї теми)")
            tm = (metrics_df[metrics_df["topic_id"] == tid]
                  .sort_values("MAE").round(3).reset_index(drop=True))
            st.dataframe(tm, width="stretch")

        if not topics_df.empty:
            with st.expander("Ключові слова / інформація про теми"):
                st.dataframe(topics_df, width="stretch")


# ============================ TRENDS TAB ===================================
with tab_trends:
    trends_df = load_trends_ts()
    tfc_df = load_trends_forecasts()
    tmetrics_df = load_trends_metrics()

    if trends_df.empty:
        st.info(
            "Немає даних Google Trends. Збери:\n\n"
            "`python -m src.collect.google_trends --timeframe 'today 5-y'`"
        )
    else:
        kws = sorted(trends_df["keyword"].unique().tolist())
        col_a, col_b = st.columns([2, 3])
        with col_a:
            kw = st.selectbox("Ключове слово", kws)
        with col_b:
            mods = sorted(tfc_df["model"].unique()) if not tfc_df.empty else []
            show_t = st.multiselect("Моделі", mods, default=mods, key="trends_models")

        kws_ts = trends_df[trends_df["keyword"] == kw].sort_values("date")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Середній інтерес", f"{kws_ts['interest'].mean():.1f}")
        c2.metric("Піковий інтерес", int(kws_ts["interest"].max()))
        c3.metric("Останній тиждень",
                  int(kws_ts["interest"].iloc[-1]) if len(kws_ts) else 0)
        recent = kws_ts.tail(8)["interest"].mean() if len(kws_ts) >= 8 else 0
        older = kws_ts.iloc[-16:-8]["interest"].mean() if len(kws_ts) >= 16 else 0
        trend_pct = ((recent - older) / older * 100.0) if older > 0 else 0.0
        c4.metric("Тренд за 8 тижнів", f"{trend_pct:+.1f}%")

        st.subheader(f"Інтерес Google Trends — {kw}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=kws_ts["date"], y=kws_ts["interest"],
            mode="lines", name="Історія",
            line=dict(color="#1f77b4", width=2),
        ))
        if not tfc_df.empty:
            sub = tfc_df[tfc_df["keyword"] == kw]
            add_forecast_traces(fig, sub, "date", "y_pred", show_t)
        fig.update_layout(height=450, xaxis_title="Дата",
                          yaxis_title="Інтерес (0–100)",
                          legend=dict(orientation="h", y=-0.2),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        if not tmetrics_df.empty:
            st.subheader("Точність прогнозу (для цього ключового слова)")
            tm = (tmetrics_df[tmetrics_df["keyword"] == kw]
                  .sort_values("MAE").round(3).reset_index(drop=True))
            st.dataframe(tm, width="stretch")


# ============================ COMPARE TAB =================================
with tab_compare:
    metrics_df = load_topic_metrics()
    tmetrics_df = load_trends_metrics()

    st.subheader("Середня точність по всіх темах LDA")
    if metrics_df.empty:
        st.info("Метрик по темах ще немає.")
    else:
        avg = (metrics_df.groupby("model")[["MAE", "RMSE", "MAPE", "sMAPE"]]
                         .mean().round(3).sort_values("MAE"))
        st.dataframe(avg, width="stretch")

        fig = go.Figure(go.Bar(
            x=avg.index, y=avg["MAE"],
            marker_color=PALETTE[:len(avg)],
            text=avg["MAE"].round(2), textposition="outside",
        ))
        fig.update_layout(title="Середній MAE по моделях (менше — краще)",
                          height=350,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Середня точність по ключових словах Google Trends")
    if tmetrics_df.empty:
        st.info("Метрик по Google Trends ще немає.")
    else:
        avg_t = (tmetrics_df.groupby("model")[["MAE", "RMSE", "MAPE", "sMAPE"]]
                            .mean().round(3).sort_values("MAE"))
        st.dataframe(avg_t, width="stretch")

        fig = go.Figure(go.Bar(
            x=avg_t.index, y=avg_t["MAE"],
            marker_color=PALETTE[:len(avg_t)],
            text=avg_t["MAE"].round(2), textposition="outside",
        ))
        fig.update_layout(title="Середній MAE по моделях (Trends)",
                          height=350,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")


# ============================ BACKTEST TAB =================================
with tab_backtest:
    st.markdown(
        "**Ретроспективна валідація.** Прогон системи в режимі «машини часу» "
        "по 12 контрольних точках за останні 12 місяців. Для кожної точки "
        "система працювала так, ніби «сьогодні» — це той день, і не бачила "
        "майбутніх даних. Потім ми порівняли її детекції з тим, що насправді "
        "сталося у наступні 8 тижнів."
    )

    summary = load_backtest_summary()
    pak = load_backtest_pak()
    bt_topics = load_backtest_topics()
    bt_trends = load_backtest_trends()
    bt_models = load_backtest_models()
    cases = load_case_index()

    if summary.empty and bt_topics.empty:
        st.warning(
            "Дані ретроспективи ще не згенеровано. Запустіть:\n\n"
            "```\n"
            "python -m src.eval.backtest --start 2025-05-01 --end 2026-04-01 --step monthly\n"
            "python -m src.eval.metrics\n"
            "python -m src.eval.case_studies\n"
            "```"
        )
    else:
        st.subheader("📊 Зведені метрики")
        if not summary.empty:
            cols = st.columns(min(len(summary.columns), 6))
            for i, col_name in enumerate(summary.columns[:6]):
                val = summary.iloc[0][col_name]
                cols[i].metric(col_name.replace("_", " ").title(),
                               f"{val:.2f}" if isinstance(val, float) else str(val))

        # ---- precision@K timeline -----------------------------------------
        if not pak.empty:
            st.subheader("🎯 Precision@K по контрольних точках")
            st.caption(
                "Для кожної дати беремо топ-K тем зі статусом Rising "
                "(відсортованих за прогнозом), і дивимося скільки з них "
                "реально виросли в наступні 8 тижнів. Що ближче до 1.0 — то "
                "точніший детектор."
            )
            fig = go.Figure()
            for k_val in sorted(pak["K"].unique()):
                sub = pak[pak["K"] == k_val].sort_values("as_of")
                fig.add_trace(go.Scatter(
                    x=sub["as_of"], y=sub["precision"],
                    mode="lines+markers", name=f"K={int(k_val)}",
                ))
            fig.update_layout(
                yaxis=dict(title="Precision", range=[0, 1.05]),
                xaxis=dict(title="as_of"),
                height=380, margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, width="stretch")

        # ---- model comparison --------------------------------------------
        if not bt_models.empty:
            st.subheader("⚖️ Порівняння моделей прогнозу (rolling-origin)")
            st.caption("Усереднені метрики по 12 контрольних точках × N тем.")
            for kind in bt_models["series_kind"].unique():
                sub = bt_models[bt_models["series_kind"] == kind].copy()
                sub = sub.sort_values("MAE")
                st.markdown(f"**{kind.title()}**")
                st.dataframe(sub.reset_index(drop=True), width="stretch")
                fig = go.Figure(go.Bar(
                    x=sub["model"], y=sub["MAE"],
                    marker_color=PALETTE[:len(sub)],
                    text=sub["MAE"].round(2), textposition="outside",
                ))
                fig.update_layout(
                    title=f"MAE по моделях — {kind}",
                    height=320, margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, width="stretch")

        # ---- lead time ----------------------------------------------------
        lt_sum = load_lead_time_summary()
        lt_hist = load_lead_time_hist()
        if not lt_sum.empty:
            st.subheader("⏱️ Лід-тайм: за скільки тижнів детектор сигналізує до піка")
            st.caption(
                "Серед детекцій типу Rising, які реально вистрелили — на якому "
                "тижні в межах 8-тижневого горизонту знаходився пік. Що більше "
                "тижнів — то раніше стейкхолдери отримують попередження."
            )
            cols = st.columns(min(len(lt_sum), 2))
            for i, (_, r) in enumerate(lt_sum.iterrows()):
                with cols[i % len(cols)]:
                    st.metric(
                        f"Канал: {r['channel']}",
                        f"{r['mean_weeks_to_peak']:.1f} тиж",
                        f"медіана {r['median_weeks_to_peak']:.0f} • "
                        f"≥4 тиж форы у {r['early_warning_pct']}% детекцій",
                    )
            if not lt_hist.empty:
                fig = go.Figure(go.Bar(
                    x=lt_hist["weeks_to_peak"], y=lt_hist["n_hits"],
                    marker_color=PALETTE[1],
                    text=lt_hist["n_hits"], textposition="outside",
                ))
                fig.update_layout(
                    title="Розподіл лід-тайму (теми)",
                    xaxis_title="Тижнів до піка",
                    yaxis_title="К-сть успішних детекцій",
                    height=300, margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, width="stretch")

        # ---- baseline comparison ------------------------------------------
        cmp_df = load_baseline_compare()
        if not cmp_df.empty:
            st.subheader("🆚 Детектор vs наївний baseline (top-K за останні 4 тижні)")
            st.caption(
                "Питання: чи дійсно складна модель краща за просту евристику "
                "«візьми те, що було найпопулярніше минулого тижня»? "
                "lift = precision детектора / precision baseline. "
                "abs_gain_pp — приріст у відсоткових пунктах."
            )
            st.dataframe(cmp_df, width="stretch", hide_index=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Детектор", x=cmp_df["K"],
                y=cmp_df["detector_precision"], marker_color=PALETTE[0],
                text=cmp_df["detector_precision"].round(2), textposition="outside",
            ))
            fig.add_trace(go.Bar(
                name="Naive top-K", x=cmp_df["K"],
                y=cmp_df["baseline_precision"], marker_color=PALETTE[2],
                text=cmp_df["baseline_precision"].round(2), textposition="outside",
            ))
            fig.update_layout(
                barmode="group",
                xaxis_title="K", yaxis_title="Precision@K",
                yaxis=dict(range=[0, 1.05]),
                height=340, margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, width="stretch")

        # ---- detections explorer ------------------------------------------
        st.subheader("🔍 Усі детекції")
        st.caption(
            "Кожен рядок — статус однієї теми/ключового слова в одну з "
            "контрольних точок, з фактичним приростом у наступні 8 тижнів."
        )

        det_view = pd.DataFrame()
        if not bt_topics.empty:
            t = bt_topics.copy()
            t["channel"] = "topic"
            t["label"] = t["keywords"].fillna("").str.slice(0, 50)
            det_view = pd.concat([det_view, t], ignore_index=True)
        if not bt_trends.empty:
            tr = bt_trends.copy()
            tr["channel"] = "trend"
            tr["label"] = tr["keyword"]
            tr["ts_growth"] = tr.get("gt_growth")
            det_view = pd.concat([det_view, tr], ignore_index=True)

        if not det_view.empty:
            available_dates = sorted(det_view["as_of"].dt.date.unique())
            sel_date = st.select_slider(
                "Контрольна точка (as_of):",
                options=available_dates,
                value=available_dates[-1],
            )
            day = det_view[det_view["as_of"].dt.date == sel_date]
            display_cols = [c for c in [
                "channel", "label", "status", "momentum_pct",
                "forecast_pct", "ts_growth", "actually_grew",
            ] if c in day.columns]
            day_show = day[display_cols].sort_values(
                "forecast_pct" if "forecast_pct" in day.columns else "momentum_pct",
                ascending=False,
            ).reset_index(drop=True)
            st.dataframe(day_show, width="stretch", hide_index=True)

        # ---- case studies -------------------------------------------------
        if not cases.empty:
            st.subheader("📚 Кейс-стаді")
            st.caption(
                "Найвиразніші приклади — успішні детекції, провали, спайки. "
                "Графіки автоматично згенеровані з історичних даних."
            )
            st.dataframe(cases, width="stretch", hide_index=True)
            for _, row in cases.iterrows():
                img_path = CASES_DIR / row["image"]
                if img_path.exists():
                    st.image(str(img_path), caption=f"{row['label']} ({row['kind']})")

