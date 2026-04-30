"""Streamlit dashboard for fashion trend prediction.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import METRICS_DIR, PROCESSED_DIR, RAW_DIR

st.set_page_config(
    page_title="Fashion Trend Forecasting",
    page_icon=":dress:",
    layout="wide",
)

st.title("Fashion Trend Forecasting Dashboard")
st.caption(
    "PhD coursework - topic modeling and time series forecasting on Guardian, "
    "News API, and Google Trends."
)

PALETTE = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_topics() -> pd.DataFrame:
    for p in [PROCESSED_DIR / "bertopic_topics.csv",
              PROCESSED_DIR / "lda_topics.csv"]:
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data
def load_topic_ts() -> pd.DataFrame:
    p = PROCESSED_DIR / "topic_timeseries.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["period"] = pd.to_datetime(df["period"])
    return df


@st.cache_data
def load_topic_forecasts() -> pd.DataFrame:
    p = PROCESSED_DIR / "forecasts.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["period"] = pd.to_datetime(df["period"])
    return df


@st.cache_data
def load_topic_metrics() -> pd.DataFrame:
    p = METRICS_DIR / "forecast_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_trends_ts() -> pd.DataFrame:
    p = RAW_DIR / "google_trends.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_trends_forecasts() -> pd.DataFrame:
    p = PROCESSED_DIR / "trends_forecasts.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_trends_metrics() -> pd.DataFrame:
    p = METRICS_DIR / "trends_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_emerging_topics() -> pd.DataFrame:
    p = METRICS_DIR / "emerging_topics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_emerging_trends() -> pd.DataFrame:
    p = METRICS_DIR / "emerging_trends.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def topic_label(tid: int, topics_df: pd.DataFrame) -> str:
    if topics_df.empty:
        return f"Topic {tid}"
    if "Topic" in topics_df.columns:  # BERTopic
        row = topics_df[topics_df["Topic"] == tid]
        if not row.empty:
            return f"{tid} - {row.iloc[0].get('Name', '')}"
    if "topic_id" in topics_df.columns:  # LDA
        row = topics_df[topics_df["topic_id"] == tid]
        if not row.empty:
            kw = str(row.iloc[0].get("keywords", ""))[:60]
            return f"{tid} - {kw}"
    return f"Topic {tid}"


def add_forecast_traces(fig, fc_df, x_col, y_col, models):
    for i, (model, grp) in enumerate(fc_df.groupby("model")):
        if model not in models:
            continue
        fig.add_trace(go.Scatter(
            x=grp[x_col], y=grp[y_col],
            mode="lines+markers", name=f"{model} (pred)",
            line=dict(color=PALETTE[i % len(PALETTE)], dash="dash"),
        ))


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_emerging, tab_topics, tab_trends, tab_compare = st.tabs(
    ["Emerging Trends", "LDA Topics", "Google Trends", "Model Comparison"]
)


# ============================ EMERGING TAB =================================
def _status_color(s: str) -> str:
    return {"Rising": "#2ca02c", "Declining": "#d62728",
            "Stable": "#7f7f7f"}.get(s, "#7f7f7f")


with tab_emerging:
    st.markdown(
        "**What's trending up vs down?** Combines historical momentum "
        "(last 8 weeks vs prior 26) with the best-model forecast for the "
        "next 8 weeks."
    )

    em_topics = load_emerging_topics()
    em_trends = load_emerging_trends()

    st.subheader("Google Trends keywords")
    if em_trends.empty:
        st.info("Run `python -m src.analysis.emerging` first.")
    else:
        cols = st.columns(len(em_trends))
        for col, (_, r) in zip(cols, em_trends.iterrows()):
            with col:
                col.markdown(f"**{r['keyword']}**")
                col.metric("Momentum (recent vs past)",
                           f"{r['momentum_pct']:+.1f}%")
                if pd.notna(r["forecast_pct"]):
                    col.metric("Forecast next 8w",
                               f"{r['forecast_pct']:+.1f}%")
                col.markdown(
                    f"<span style='background:{_status_color(r['status'])};"
                    f"color:white;padding:4px 10px;border-radius:6px;'>"
                    f"{r['status']}</span>",
                    unsafe_allow_html=True,
                )

        fig = go.Figure()
        em_sorted = em_trends.sort_values("momentum_pct")
        fig.add_trace(go.Bar(
            x=em_sorted["momentum_pct"], y=em_sorted["keyword"],
            orientation="h", name="Historical momentum",
            marker_color=[_status_color(s) for s in em_sorted["status"]],
            text=[f"{v:+.0f}%" for v in em_sorted["momentum_pct"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=300, xaxis_title="Momentum, %",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

        st.dataframe(em_trends, width="stretch")

    st.subheader("LDA topics ranked by forecast growth")
    if em_topics.empty:
        st.info("Run `python -m src.analysis.emerging` first.")
    else:
        rising = em_topics[em_topics["status"] == "Rising"]
        declining = em_topics[em_topics["status"] == "Declining"]
        stable = em_topics[em_topics["status"] == "Stable"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Rising topics", len(rising))
        c2.metric("Stable topics", len(stable))
        c3.metric("Declining topics", len(declining))

        st.markdown("##### Top 10 by historical momentum")
        top = em_topics.sort_values("momentum_pct", ascending=False).head(10)
        fig = go.Figure(go.Bar(
            x=top["momentum_pct"],
            y=[f"#{int(t)} - {str(k)[:55]}" for t, k in
               zip(top["topic_id"], top["keywords"])],
            orientation="h",
            marker_color=[_status_color(s) for s in top["status"]],
            text=[f"{v:+.0f}%" for v in top["momentum_pct"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=420, xaxis_title="Momentum, %",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("##### Full ranking")
        st.dataframe(em_topics, width="stretch")

        st.caption(
            "**How to read:** *momentum_pct* tells you what is gaining "
            "attention RIGHT NOW (last 8 weeks vs prior 26). "
            "*forecast_pct* is the model's projection for the next 8 weeks "
            "(positive = the model expects continued growth). "
            "Topics with high momentum but negative forecast are likely "
            "near peak hype."
        )


# ============================ TOPICS TAB ===================================
with tab_topics:
    topics_df = load_topics()
    ts_df = load_topic_ts()
    fc_df = load_topic_forecasts()
    metrics_df = load_topic_metrics()

    if ts_df.empty:
        st.warning("No topic time series found. Run `python -m src.run_pipeline`.")
    else:
        col_a, col_b = st.columns([2, 3])
        with col_a:
            tids = sorted(ts_df["topic_id"].unique().tolist())
            tid = st.selectbox("Topic", tids,
                               format_func=lambda t: topic_label(t, topics_df))
        with col_b:
            available_models = sorted(fc_df["model"].unique()) if not fc_df.empty else []
            show = st.multiselect("Models to display", available_models,
                                  default=available_models)

        topic_ts = ts_df[ts_df["topic_id"] == tid].sort_values("period")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total mentions", int(topic_ts["count"].sum()))
        c2.metric("Peak weekly count", int(topic_ts["count"].max()))
        c3.metric("Latest week",
                  int(topic_ts["count"].iloc[-1]) if len(topic_ts) else 0)
        recent = topic_ts.tail(8)["count"].mean() if len(topic_ts) >= 8 else 0
        older = topic_ts.iloc[-16:-8]["count"].mean() if len(topic_ts) >= 16 else 0
        trend_pct = ((recent - older) / older * 100.0) if older > 0 else 0.0
        c4.metric("8-week trend", f"{trend_pct:+.1f}%")

        st.subheader(f"Time series - {topic_label(tid, topics_df)}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=topic_ts["period"], y=topic_ts["count"],
            mode="lines", name="History",
            line=dict(color="#1f77b4", width=2),
        ))
        if not fc_df.empty:
            sub = fc_df[fc_df["topic_id"] == tid]
            add_forecast_traces(fig, sub, "period", "y_pred", show)
        fig.update_layout(height=450, xaxis_title="Date",
                          yaxis_title="Posts per week",
                          legend=dict(orientation="h", y=-0.2),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        if not metrics_df.empty:
            st.subheader("Forecast accuracy (this topic)")
            tm = (metrics_df[metrics_df["topic_id"] == tid]
                  .sort_values("MAE").round(3).reset_index(drop=True))
            st.dataframe(tm, width="stretch")

        if not topics_df.empty:
            with st.expander("Topic keywords / info"):
                st.dataframe(topics_df, width="stretch")


# ============================ TRENDS TAB ===================================
with tab_trends:
    trends_df = load_trends_ts()
    tfc_df = load_trends_forecasts()
    tmetrics_df = load_trends_metrics()

    if trends_df.empty:
        st.info(
            "No Google Trends data. Collect with:\n\n"
            "`python -m src.collect.google_trends --timeframe 'today 5-y'`"
        )
    else:
        kws = sorted(trends_df["keyword"].unique().tolist())
        col_a, col_b = st.columns([2, 3])
        with col_a:
            kw = st.selectbox("Keyword", kws)
        with col_b:
            mods = sorted(tfc_df["model"].unique()) if not tfc_df.empty else []
            show_t = st.multiselect("Models", mods, default=mods, key="trends_models")

        kws_ts = trends_df[trends_df["keyword"] == kw].sort_values("date")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean interest", f"{kws_ts['interest'].mean():.1f}")
        c2.metric("Peak interest", int(kws_ts["interest"].max()))
        c3.metric("Latest week",
                  int(kws_ts["interest"].iloc[-1]) if len(kws_ts) else 0)
        recent = kws_ts.tail(8)["interest"].mean() if len(kws_ts) >= 8 else 0
        older = kws_ts.iloc[-16:-8]["interest"].mean() if len(kws_ts) >= 16 else 0
        trend_pct = ((recent - older) / older * 100.0) if older > 0 else 0.0
        c4.metric("8-week trend", f"{trend_pct:+.1f}%")

        st.subheader(f"Google Trends interest - {kw}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=kws_ts["date"], y=kws_ts["interest"],
            mode="lines", name="History",
            line=dict(color="#1f77b4", width=2),
        ))
        if not tfc_df.empty:
            sub = tfc_df[tfc_df["keyword"] == kw]
            add_forecast_traces(fig, sub, "date", "y_pred", show_t)
        fig.update_layout(height=450, xaxis_title="Date",
                          yaxis_title="Interest (0-100)",
                          legend=dict(orientation="h", y=-0.2),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        if not tmetrics_df.empty:
            st.subheader("Forecast accuracy (this keyword)")
            tm = (tmetrics_df[tmetrics_df["keyword"] == kw]
                  .sort_values("MAE").round(3).reset_index(drop=True))
            st.dataframe(tm, width="stretch")


# ============================ COMPARE TAB =================================
with tab_compare:
    metrics_df = load_topic_metrics()
    tmetrics_df = load_trends_metrics()

    st.subheader("Average performance across LDA topics")
    if metrics_df.empty:
        st.info("No topic metrics yet.")
    else:
        avg = (metrics_df.groupby("model")[["MAE", "RMSE", "MAPE", "sMAPE"]]
                         .mean().round(3).sort_values("MAE"))
        st.dataframe(avg, width="stretch")

        fig = go.Figure(go.Bar(
            x=avg.index, y=avg["MAE"],
            marker_color=PALETTE[:len(avg)],
            text=avg["MAE"].round(2), textposition="outside",
        ))
        fig.update_layout(title="Mean MAE by model (lower = better)",
                          height=350,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Average performance on Google Trends keywords")
    if tmetrics_df.empty:
        st.info("No Google Trends metrics yet.")
    else:
        avg_t = (tmetrics_df.groupby("model")[["MAE", "RMSE", "MAPE", "sMAPE"]]
                            .mean().round(3).sort_values("MAE"))
        st.dataframe(avg_t, width="stretch")

        fig = go.Figure(go.Bar(
            x=avg_t.index, y=avg_t["MAE"],
            marker_color=PALETTE[:len(avg_t)],
            text=avg_t["MAE"].round(2), textposition="outside",
        ))
        fig.update_layout(title="Mean MAE by model on Trends",
                          height=350,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
