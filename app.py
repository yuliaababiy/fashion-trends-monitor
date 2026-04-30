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
    page_title="РџСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ РјРѕРґРЅРёС… С‚СЂРµРЅРґС–РІ",
    page_icon=":dress:",
    layout="wide",
)

st.title("рџ‘— РџСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ РјРѕРґРЅРёС… С‚СЂРµРЅРґС–РІ Сѓ СЃРѕС†С–Р°Р»СЊРЅРёС… РјРµСЂРµР¶Р°С…")
st.caption(
    "РљСѓСЂСЃРѕРІР° СЂРѕР±РѕС‚Р° вЂ” С‚РµРјР°С‚РёС‡РЅРµ РјРѕРґРµР»СЋРІР°РЅРЅСЏ С‚Р° РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ С‡Р°СЃРѕРІРёС… СЂСЏРґС–РІ РЅР° РѕСЃРЅРѕРІС– "
    "The Guardian, News API С‚Р° Google Trends."
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def topic_label(tid: int, topics_df: pd.DataFrame) -> str:
    if topics_df.empty:
        return f"РўРµРјР° {tid}"
    if "Topic" in topics_df.columns:  # BERTopic
        row = topics_df[topics_df["Topic"] == tid]
        if not row.empty:
            return f"{tid} - {row.iloc[0].get('Name', '')}"
    if "topic_id" in topics_df.columns:  # LDA
        row = topics_df[topics_df["topic_id"] == tid]
        if not row.empty:
            kw = str(row.iloc[0].get("keywords", ""))[:60]
            return f"{tid} - {kw}"
    return f"РўРµРјР° {tid}"


def add_forecast_traces(fig, fc_df, x_col, y_col, models):
    for i, (model, grp) in enumerate(fc_df.groupby("model")):
        if model not in models:
            continue
        fig.add_trace(go.Scatter(
            x=grp[x_col], y=grp[y_col],
            mode="lines+markers", name=f"{model} (РїСЂРѕРіРЅРѕР·)",
            line=dict(color=PALETTE[i % len(PALETTE)], dash="dash"),
        ))


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_emerging, tab_spikes, tab_topics, tab_trends, tab_compare = st.tabs(
    ["рџ“€ РўСЂРµРЅРґРё, С‰Рѕ Р·СЂРѕСЃС‚Р°СЋС‚СЊ", "рџ”Ґ РЎРїР»РµСЃРєРё СЃР»С–РІ",
     "рџ”Ќ РўРµРјРё LDA", "рџЊђ Google Trends", "вљ–пёЏ РџРѕСЂС–РІРЅСЏРЅРЅСЏ РјРѕРґРµР»РµР№"]
)


# ============================ EMERGING TAB =================================
def _status_color(s: str) -> str:
    return {"Rising": "#2ca02c", "Declining": "#d62728",
            "Stable": "#7f7f7f"}.get(s, "#7f7f7f")


STATUS_UA = {"Rising": "Р·СЂРѕСЃС‚Р°С”", "Declining": "СЃРїР°РґР°С”", "Stable": "СЃС‚Р°Р±С–Р»СЊРЅРёР№"}


with tab_emerging:
    st.markdown(
        "**Р©Рѕ Р·СЂРѕСЃС‚Р°С”, Р° С‰Рѕ вЂ” СЃРїР°РґР°С”?** РџРѕС”РґРЅСѓС” С–СЃС‚РѕСЂРёС‡РЅРёР№ РјРѕРјРµРЅС‚СѓРј "
        "(РѕСЃС‚Р°РЅРЅС– 8 С‚РёР¶РЅС–РІ vs РїРѕРїРµСЂРµРґРЅС– 26) Р· РїСЂРѕРіРЅРѕР·РѕРј РЅР°Р№РєСЂР°С‰РѕС— РјРѕРґРµР»С– "
        "РЅР° РЅР°СЃС‚СѓРїРЅС– 8 С‚РёР¶РЅС–РІ."
    )

    em_topics = load_emerging_topics()
    em_trends = load_emerging_trends()

    st.subheader("РљР»СЋС‡РѕРІС– СЃР»РѕРІР° Google Trends")
    if em_trends.empty:
        st.info("РЎРїРѕС‡Р°С‚РєСѓ РІРёРєРѕРЅР°Р№С‚Рµ `python -m src.analysis.emerging`.")
    else:
        cols = st.columns(len(em_trends))
        for col, (_, r) in zip(cols, em_trends.iterrows()):
            with col:
                col.markdown(f"**{r['keyword']}**")
                col.metric("РњРѕРјРµРЅС‚СѓРј (РѕСЃС‚Р°РЅРЅС– vs РїРѕРїРµСЂРµРґРЅС–)",
                           f"{r['momentum_pct']:+.1f}%")
                if pd.notna(r["forecast_pct"]):
                    col.metric("РџСЂРѕРіРЅРѕР· РЅР° 8 С‚РёР¶РЅС–РІ",
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
            orientation="h", name="Р†СЃС‚РѕСЂРёС‡РЅРёР№ РјРѕРјРµРЅС‚СѓРј",
            marker_color=[_status_color(s) for s in em_sorted["status"]],
            text=[f"{v:+.0f}%" for v in em_sorted["momentum_pct"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=300, xaxis_title="РњРѕРјРµРЅС‚СѓРј, %",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

        st.dataframe(em_trends, width="stretch")

    st.subheader("РўРµРјРё LDA, РІРїРѕСЂСЏРґРєРѕРІР°РЅС– Р·Р° РїСЂРѕРіРЅРѕР·РѕРј Р·СЂРѕСЃС‚Р°РЅРЅСЏ")
    if em_topics.empty:
        st.info("РЎРїРѕС‡Р°С‚РєСѓ РІРёРєРѕРЅР°Р№С‚Рµ `python -m src.analysis.emerging`.")
    else:
        rising = em_topics[em_topics["status"] == "Rising"]
        declining = em_topics[em_topics["status"] == "Declining"]
        stable = em_topics[em_topics["status"] == "Stable"]

        c1, c2, c3 = st.columns(3)
        c1.metric("РўРµРј, С‰Рѕ Р·СЂРѕСЃС‚Р°СЋС‚СЊ", len(rising))
        c2.metric("РЎС‚Р°Р±С–Р»СЊРЅРёС… С‚РµРј", len(stable))
        c3.metric("РўРµРј, С‰Рѕ СЃРїР°РґР°СЋС‚СЊ", len(declining))

        st.markdown("##### РўРѕРї-10 Р·Р° С–СЃС‚РѕСЂРёС‡РЅРёРј РјРѕРјРµРЅС‚СѓРјРѕРј")
        top = em_topics.sort_values("momentum_pct", ascending=False).head(10)
        labels = [f"#{int(t)} вЂ” {str(k)[:90]}" for t, k in
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
            height=520, xaxis_title="РњРѕРјРµРЅС‚СѓРј, %",
            yaxis=dict(autorange="reversed", automargin=True),
            margin=dict(l=10, r=40, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("##### РџРѕРІРЅРёР№ СЂРµР№С‚РёРЅРі")
        st.dataframe(
            em_topics,
            width="stretch",
            column_config={
                "keywords": st.column_config.TextColumn(
                    "keywords", width="large"),
            },
        )

        st.caption(
            "**РЇРє С‡РёС‚Р°С‚Рё:** *momentum_pct* вЂ” С‰Рѕ РїСЂРёРІРµСЂС‚Р°С” СѓРІР°РіСѓ РЎРђРњР• Р—РђР РђР— "
            "(РѕСЃС‚Р°РЅРЅС– 8 С‚РёР¶РЅС–РІ vs РїРѕРїРµСЂРµРґРЅС– 26). "
            "*forecast_pct* вЂ” РїСЂРѕРіРЅРѕР· РјРѕРґРµР»С– РЅР° РЅР°СЃС‚СѓРїРЅС– 8 С‚РёР¶РЅС–РІ "
            "(РґРѕРґР°С‚РЅС” Р·РЅР°С‡РµРЅРЅСЏ вЂ” РѕС‡С–РєСѓС”С‚СЊСЃСЏ РїРѕРґР°Р»СЊС€Рµ Р·СЂРѕСЃС‚Р°РЅРЅСЏ). "
            "РўРµРјРё Р· РІРёСЃРѕРєРёРј РјРѕРјРµРЅС‚СѓРјРѕРј, Р°Р»Рµ РІС–Рґ'С”РјРЅРёРј РїСЂРѕРіРЅРѕР·РѕРј вЂ” Р№РјРѕРІС–СЂРЅРѕ Р±Р»РёР·СЊРєРѕ РґРѕ РїС–РєСѓ."
        )


# ============================ SPIKES TAB ===================================
with tab_spikes:
    st.markdown(
        "**РЎРїР»РµСЃРєРё РѕРєСЂРµРјРёС… СЃР»С–РІ (TF-IDF).** РЁСѓРєР°С”РјРѕ СЃР»РѕРІР°, С‡Р°СЃС‚РѕС‚Р° СЏРєРёС… "
        "Р·Р° РѕСЃС‚Р°РЅРЅС–Р№ С‚РёР¶РґРµРЅСЊ Р°РЅРѕРјР°Р»СЊРЅРѕ РІРёС‰Р° Р·Р° Р±Р°Р·Сѓ РїРѕРїРµСЂРµРґРЅС–С… 12 С‚РёР¶РЅС–РІ. "
        "Р¦Рµ СЂР°РЅРЅС” РїРѕРїРµСЂРµРґР¶РµРЅРЅСЏ вЂ” СЃР»РѕРІР° РјРѕР¶СѓС‚СЊ Р·'СЏРІРёС‚РёСЃСЏ С‰Рµ РґРѕ С‚РѕРіРѕ, СЏРє "
        "СЃС„РѕСЂРјСѓСЋС‚СЊ РїРѕРІРЅРѕС†С–РЅРЅСѓ С‚РµРјСѓ LDA."
    )

    spikes = load_spike_terms()
    if spikes.empty:
        st.info(
            "РџРѕРєРё РЅРµРјР°С” `spike_terms.csv`. Р—Р°РїСѓСЃС‚С–С‚СЊ "
            "`python -m src.analysis.spikes` Р°Р±Рѕ Р·Р°С‡РµРєР°Р№С‚Рµ РЅР° С‰РѕРґРµРЅРЅРёР№ РїР°Р№РїР»Р°Р№РЅ."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Р’РёСЏРІР»РµРЅРѕ СЃРїР»РµСЃРєС–РІ", len(spikes))
        c2.metric("РќР°Р№РІРёС‰РёР№ z-score", f"{spikes['z_score'].max():.1f}")
        c3.metric("РќР°Р№Р±С–Р»СЊС€РёР№ Г—ratio", f"{spikes['spike_ratio'].max():.1f}")

        st.markdown("##### РўРѕРї-15 Р·Р° z-score")
        top = spikes.sort_values("z_score", ascending=False).head(15)
        fig = go.Figure(go.Bar(
            x=top["z_score"],
            y=top["term"],
            orientation="h",
            marker_color="#d62728",
            text=[f"Г—{r:.1f}" for r in top["spike_ratio"]],
            textposition="outside",
            hovertext=[
                f"{t}: recent={int(rc)}, base={bm:.1f}В±{bs:.1f}"
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

        st.markdown("##### РџРѕРІРЅР° С‚Р°Р±Р»РёС†СЏ")
        st.dataframe(spikes, width="stretch")
        st.caption(
            "**z-score** вЂ” РЅР°СЃРєС–Р»СЊРєРё РѕСЃС‚Р°РЅРЅС–Р№ С‚РёР¶РґРµРЅСЊ РІС–РґС…РёР»СЏС”С‚СЊСЃСЏ РІС–Рґ "
            "СЃРµСЂРµРґРЅСЊРѕРіРѕ Р±Р°Р·Рё (в‰Ґ3 = Р°РЅРѕРјР°Р»С–СЏ). **spike_ratio** вЂ” Сѓ СЃРєС–Р»СЊРєРё "
            "СЂР°Р·С–РІ С‡Р°СЃС‚С–С€Рµ СЃР»РѕРІРѕ Р·РіР°РґСѓРІР°Р»РѕСЃСЊ Р·Р°СЂР°Р· РїРѕСЂС–РІРЅСЏРЅРѕ Р· Р±Р°Р·РѕСЋ. "
            "**recent_count** вЂ” Р·РіР°РґРѕРє Р·Р° РѕСЃС‚Р°РЅРЅС–Р№ С‚РёР¶РґРµРЅСЊ."
        )


# ============================ TOPICS TAB ===================================
with tab_topics:
    topics_df = load_topics()
    ts_df = load_topic_ts()
    fc_df = load_topic_forecasts()
    metrics_df = load_topic_metrics()

    if ts_df.empty:
        st.warning("РќРµРјР°С” С‡Р°СЃРѕРІРёС… СЂСЏРґС–РІ С‚РµРј. Р—Р°РїСѓСЃС‚С–С‚СЊ `python -m src.run_pipeline`.")
    else:
        col_a, col_b = st.columns([2, 3])
        with col_a:
            tids = sorted(ts_df["topic_id"].unique().tolist())
            tid = st.selectbox("РўРµРјР°", tids,
                               format_func=lambda t: topic_label(t, topics_df))
        with col_b:
            available_models = sorted(fc_df["model"].unique()) if not fc_df.empty else []
            show = st.multiselect("РњРѕРґРµР»С– РґР»СЏ РІС–РґРѕР±СЂР°Р¶РµРЅРЅСЏ", available_models,
                                  default=available_models)

        topic_ts = ts_df[ts_df["topic_id"] == tid].sort_values("period")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Р’СЃСЊРѕРіРѕ Р·РіР°РґРѕРє", int(topic_ts["count"].sum()))
        c2.metric("РџС–Рє Р·Р° С‚РёР¶РґРµРЅСЊ", int(topic_ts["count"].max()))
        c3.metric("РћСЃС‚Р°РЅРЅС–Р№ С‚РёР¶РґРµРЅСЊ",
                  int(topic_ts["count"].iloc[-1]) if len(topic_ts) else 0)
        recent = topic_ts.tail(8)["count"].mean() if len(topic_ts) >= 8 else 0
        older = topic_ts.iloc[-16:-8]["count"].mean() if len(topic_ts) >= 16 else 0
        trend_pct = ((recent - older) / older * 100.0) if older > 0 else 0.0
        c4.metric("РўСЂРµРЅРґ Р·Р° 8 С‚РёР¶РЅС–РІ", f"{trend_pct:+.1f}%")

        st.subheader(f"Р§Р°СЃРѕРІРёР№ СЂСЏРґ вЂ” {topic_label(tid, topics_df)}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=topic_ts["period"], y=topic_ts["count"],
            mode="lines", name="Р†СЃС‚РѕСЂС–СЏ",
            line=dict(color="#1f77b4", width=2),
        ))
        if not fc_df.empty:
            sub = fc_df[fc_df["topic_id"] == tid]
            add_forecast_traces(fig, sub, "period", "y_pred", show)
        fig.update_layout(height=450, xaxis_title="Р”Р°С‚Р°",
                          yaxis_title="РџСѓР±Р»С–РєР°С†С–Р№ РЅР° С‚РёР¶РґРµРЅСЊ",
                          legend=dict(orientation="h", y=-0.2),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        if not metrics_df.empty:
            st.subheader("РўРѕС‡РЅС–СЃС‚СЊ РїСЂРѕРіРЅРѕР·Сѓ (РґР»СЏ С†С–С”С— С‚РµРјРё)")
            tm = (metrics_df[metrics_df["topic_id"] == tid]
                  .sort_values("MAE").round(3).reset_index(drop=True))
            st.dataframe(tm, width="stretch")

        if not topics_df.empty:
            with st.expander("РљР»СЋС‡РѕРІС– СЃР»РѕРІР° / С–РЅС„РѕСЂРјР°С†С–СЏ РїСЂРѕ С‚РµРјРё"):
                st.dataframe(topics_df, width="stretch")


# ============================ TRENDS TAB ===================================
with tab_trends:
    trends_df = load_trends_ts()
    tfc_df = load_trends_forecasts()
    tmetrics_df = load_trends_metrics()

    if trends_df.empty:
        st.info(
            "РќРµРјР°С” РґР°РЅРёС… Google Trends. Р—Р±РµСЂРё:\n\n"
            "`python -m src.collect.google_trends --timeframe 'today 5-y'`"
        )
    else:
        kws = sorted(trends_df["keyword"].unique().tolist())
        col_a, col_b = st.columns([2, 3])
        with col_a:
            kw = st.selectbox("РљР»СЋС‡РѕРІРµ СЃР»РѕРІРѕ", kws)
        with col_b:
            mods = sorted(tfc_df["model"].unique()) if not tfc_df.empty else []
            show_t = st.multiselect("РњРѕРґРµР»С–", mods, default=mods, key="trends_models")

        kws_ts = trends_df[trends_df["keyword"] == kw].sort_values("date")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("РЎРµСЂРµРґРЅС–Р№ С–РЅС‚РµСЂРµСЃ", f"{kws_ts['interest'].mean():.1f}")
        c2.metric("РџС–РєРѕРІРёР№ С–РЅС‚РµСЂРµСЃ", int(kws_ts["interest"].max()))
        c3.metric("РћСЃС‚Р°РЅРЅС–Р№ С‚РёР¶РґРµРЅСЊ",
                  int(kws_ts["interest"].iloc[-1]) if len(kws_ts) else 0)
        recent = kws_ts.tail(8)["interest"].mean() if len(kws_ts) >= 8 else 0
        older = kws_ts.iloc[-16:-8]["interest"].mean() if len(kws_ts) >= 16 else 0
        trend_pct = ((recent - older) / older * 100.0) if older > 0 else 0.0
        c4.metric("РўСЂРµРЅРґ Р·Р° 8 С‚РёР¶РЅС–РІ", f"{trend_pct:+.1f}%")

        st.subheader(f"Р†РЅС‚РµСЂРµСЃ Google Trends вЂ” {kw}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=kws_ts["date"], y=kws_ts["interest"],
            mode="lines", name="Р†СЃС‚РѕСЂС–СЏ",
            line=dict(color="#1f77b4", width=2),
        ))
        if not tfc_df.empty:
            sub = tfc_df[tfc_df["keyword"] == kw]
            add_forecast_traces(fig, sub, "date", "y_pred", show_t)
        fig.update_layout(height=450, xaxis_title="Р”Р°С‚Р°",
                          yaxis_title="Р†РЅС‚РµСЂРµСЃ (0вЂ“100)",
                          legend=dict(orientation="h", y=-0.2),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        if not tmetrics_df.empty:
            st.subheader("РўРѕС‡РЅС–СЃС‚СЊ РїСЂРѕРіРЅРѕР·Сѓ (РґР»СЏ С†СЊРѕРіРѕ РєР»СЋС‡РѕРІРѕРіРѕ СЃР»РѕРІР°)")
            tm = (tmetrics_df[tmetrics_df["keyword"] == kw]
                  .sort_values("MAE").round(3).reset_index(drop=True))
            st.dataframe(tm, width="stretch")


# ============================ COMPARE TAB =================================
with tab_compare:
    metrics_df = load_topic_metrics()
    tmetrics_df = load_trends_metrics()

    st.subheader("РЎРµСЂРµРґРЅСЏ С‚РѕС‡РЅС–СЃС‚СЊ РїРѕ РІСЃС–С… С‚РµРјР°С… LDA")
    if metrics_df.empty:
        st.info("РњРµС‚СЂРёРє РїРѕ С‚РµРјР°С… С‰Рµ РЅРµРјР°С”.")
    else:
        avg = (metrics_df.groupby("model")[["MAE", "RMSE", "MAPE", "sMAPE"]]
                         .mean().round(3).sort_values("MAE"))
        st.dataframe(avg, width="stretch")

        fig = go.Figure(go.Bar(
            x=avg.index, y=avg["MAE"],
            marker_color=PALETTE[:len(avg)],
            text=avg["MAE"].round(2), textposition="outside",
        ))
        fig.update_layout(title="РЎРµСЂРµРґРЅС–Р№ MAE РїРѕ РјРѕРґРµР»СЏС… (РјРµРЅС€Рµ вЂ” РєСЂР°С‰Рµ)",
                          height=350,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("РЎРµСЂРµРґРЅСЏ С‚РѕС‡РЅС–СЃС‚СЊ РїРѕ РєР»СЋС‡РѕРІРёС… СЃР»РѕРІР°С… Google Trends")
    if tmetrics_df.empty:
        st.info("РњРµС‚СЂРёРє РїРѕ Google Trends С‰Рµ РЅРµРјР°С”.")
    else:
        avg_t = (tmetrics_df.groupby("model")[["MAE", "RMSE", "MAPE", "sMAPE"]]
                            .mean().round(3).sort_values("MAE"))
        st.dataframe(avg_t, width="stretch")

        fig = go.Figure(go.Bar(
            x=avg_t.index, y=avg_t["MAE"],
            marker_color=PALETTE[:len(avg_t)],
            text=avg_t["MAE"].round(2), textposition="outside",
        ))
        fig.update_layout(title="РЎРµСЂРµРґРЅС–Р№ MAE РїРѕ РјРѕРґРµР»СЏС… (Trends)",
                          height=350,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

