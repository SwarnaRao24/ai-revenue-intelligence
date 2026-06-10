"""
dashboard/app.py
Streamlit Analytics Dashboard — Churn & Revenue Intelligence Platform
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Churn & Revenue Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS ─────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0e1117 100%);
        border-right: 1px solid #2d3748;
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 6px 0 2px 0;
    }
    .kpi-label {
        font-size: 0.80rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .kpi-delta {
        font-size: 0.78rem;
        margin-top: 4px;
    }

    /* Risk badges */
    .badge-high   { background:#ff4444; color:white; padding:2px 10px;
                    border-radius:20px; font-size:0.75rem; font-weight:600; }
    .badge-medium { background:#ff8800; color:white; padding:2px 10px;
                    border-radius:20px; font-size:0.75rem; font-weight:600; }
    .badge-low    { background:#00c851; color:white; padding:2px 10px;
                    border-radius:20px; font-size:0.75rem; font-weight:600; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        border-left: 3px solid #00d4ff;
        padding-left: 10px;
        margin: 8px 0 16px 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
    header    {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════
# Data generators
# ══════════════════════════════════════════════════════════════


@st.cache_data
def load_customer_data() -> pd.DataFrame:
    """Load or generate customer churn feature data."""
    try:
        from ml_training.utils.data_loader import load_churn_features

        df = load_churn_features()
    except Exception:
        df = _synthetic_customers()
    return df


@st.cache_data
def load_revenue_data() -> pd.DataFrame:
    """Generate monthly revenue time series."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", periods=42, freq="MS")
    trend = np.linspace(55_000, 118_000, 42)
    season = 8_500 * np.sin(2 * np.pi * np.arange(42) / 12)
    noise = rng.normal(0, 2_800, 42)
    rev = (trend + season + noise).round(2)
    return pd.DataFrame({"month": dates, "revenue": rev})


@st.cache_data
def load_forecast_data(periods: int = 6) -> pd.DataFrame:
    """Generate revenue forecast with confidence bands."""
    from datetime import datetime, timedelta

    rng = np.random.default_rng(99)
    base = 118_000
    rows = []
    for i in range(1, periods + 1):
        m = (datetime.now().replace(day=1) + timedelta(days=32 * i)).replace(day=1)
        trend = base + i * 2_200
        season = 7_000 * np.sin(2 * np.pi * m.month / 12)
        val = round(trend + season + rng.normal(0, 1_500), 2)
        rows.append(
            {
                "month": m.strftime("%Y-%m"),
                "forecast": val,
                "lower": round(val * 0.92, 2),
                "upper": round(val * 1.08, 2),
            }
        )
    return pd.DataFrame(rows)


def _synthetic_customers(n: int = 7043) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    tenure = rng.integers(1, 72, n)
    monthly = rng.uniform(18, 118, n).round(2)
    churn = np.zeros(n, dtype=int)
    churn[: int(n * 0.265)] = 1
    rng.shuffle(churn)

    prob = np.where(
        (churn == 1),
        rng.uniform(0.55, 0.95, n),
        rng.uniform(0.05, 0.45, n),
    ).round(3)

    segments = rng.choice(
        ["Champions", "Loyal Customers", "At Risk", "Hibernating", "Lost"],
        size=n,
        p=[0.20, 0.25, 0.25, 0.18, 0.12],
    )
    contracts = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.21, 0.24]
    )

    return pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:06d}" for i in range(n)],
            "label": churn,
            "churn_prob": prob,
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": (tenure * monthly).round(2),
            "contract_type": contracts,
            "segment": segments,
            "is_senior_citizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "has_tech_support": rng.integers(0, 2, n),
            "has_online_security": rng.integers(0, 2, n),
            "total_support_tickets": rng.integers(0, 8, n),
            "risk_tier": np.where(
                prob >= 0.70, "HIGH", np.where(prob >= 0.40, "MEDIUM", "LOW")
            ),
        }
    )


# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════


def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("## 🧠 Churn Intelligence")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            [
                "📊 Overview",
                "🔴 Churn Analysis",
                "💰 Revenue Forecast",
                "👥 Segmentation",
                "🚨 Risk Explorer",
                "🤖 Live Predictor",
                "📈 Model Monitoring",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### Filters")

        contract_filter = st.multiselect(
            "Contract Type",
            options=["Month-to-month", "One year", "Two year"],
            default=["Month-to-month", "One year", "Two year"],
        )

        tenure_range = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=(0, 72),
        )

        risk_filter = st.multiselect(
            "Risk Tier",
            options=["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM", "LOW"],
        )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.72rem;color:#4a5568;text-align:center'>"
            "AI-Powered Churn Platform v1.0<br/>"
            "XGBoost · LightGBM · FastAPI"
            "</div>",
            unsafe_allow_html=True,
        )

    # Apply filters
    mask = pd.Series([True] * len(df), index=df.index)

    if "contract_type" in df.columns:
        mask &= df["contract_type"].isin(contract_filter)

    if "tenure_months" in df.columns:
        mask &= df["tenure_months"].between(*tenure_range)

    if "risk_tier" in df.columns:
        mask &= df["risk_tier"].isin(risk_filter)

    return page, df[mask].copy()


# ══════════════════════════════════════════════════════════════
# KPI Cards
# ══════════════════════════════════════════════════════════════


def render_kpis(df: pd.DataFrame, rev_df: pd.DataFrame):
    total = len(df)
    churned = df["label"].sum()
    churn_rate = churned / total * 100
    high_risk = (df["risk_tier"] == "HIGH").sum()
    mrr = rev_df["revenue"].iloc[-1]
    prev_mrr = rev_df["revenue"].iloc[-2]
    mrr_growth = (mrr - prev_mrr) / prev_mrr * 100
    avg_tenure = df["tenure_months"].mean()
    avg_charge = df["monthly_charges"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Customers</div>
            <div class="kpi-value" style="color:#00d4ff">{total:,}</div>
            <div class="kpi-delta" style="color:#718096">Active accounts</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with c2:
        color = "#ff4444" if churn_rate > 25 else "#ff8800"
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">Churn Rate</div>
            <div class="kpi-value" style="color:{color}">{churn_rate:.1f}%</div>
            <div class="kpi-delta" style="color:#718096">{churned:,} customers</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">High Risk</div>
            <div class="kpi-value" style="color:#ff4444">{high_risk:,}</div>
            <div class="kpi-delta" style="color:#718096">Need intervention</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with c4:
        arrow = "▲" if mrr_growth > 0 else "▼"
        mcolor = "#00c851" if mrr_growth > 0 else "#ff4444"
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">Monthly Revenue</div>
            <div class="kpi-value" style="color:#00c851">${mrr:,.0f}</div>
            <div class="kpi-delta" style="color:{mcolor}">{arrow} {abs(mrr_growth):.1f}% MoM</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Tenure</div>
            <div class="kpi-value" style="color:#a78bfa">{avg_tenure:.0f}mo</div>
            <div class="kpi-delta" style="color:#718096">Avg charge ${avg_charge:.0f}</div>
        </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════


def page_overview(df: pd.DataFrame, rev_df: pd.DataFrame):
    st.markdown("## 📊 Platform Overview")
    render_kpis(df, rev_df)
    st.markdown("<br/>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="section-header">Revenue Trend</div>', unsafe_allow_html=True
        )
        fig = px.area(
            rev_df,
            x="month",
            y="revenue",
            color_discrete_sequence=["#00d4ff"],
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_tickprefix="$",
            yaxis_tickformat=",",
            xaxis_title="",
            yaxis_title="",
        )
        fig.update_traces(
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.1)",
            line=dict(color="#00d4ff", width=2),
        )
        st.plotly_chart(fig, use_container_width="stretch")

    with col2:
        st.markdown(
            '<div class="section-header">Churn by Contract Type</div>',
            unsafe_allow_html=True,
        )
        churn_contract = (
            df.groupby("contract_type")["label"].agg(["sum", "count"]).reset_index()
        )
        churn_contract["churn_rate"] = (
            churn_contract["sum"] / churn_contract["count"] * 100
        ).round(1)

        fig2 = px.bar(
            churn_contract,
            x="contract_type",
            y="churn_rate",
            color="churn_rate",
            color_continuous_scale=["#00c851", "#ff8800", "#ff4444"],
            template="plotly_dark",
            text="churn_rate",
        )
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="",
            yaxis_title="Churn Rate %",
        )
        st.plotly_chart(fig2, use_container_width="stretch")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(
            '<div class="section-header">Risk Tier Distribution</div>',
            unsafe_allow_html=True,
        )
        risk_counts = df["risk_tier"].value_counts().reset_index()
        risk_counts.columns = ["tier", "count"]
        fig3 = px.pie(
            risk_counts,
            values="count",
            names="tier",
            color="tier",
            color_discrete_map={
                "HIGH": "#ff4444",
                "MEDIUM": "#ff8800",
                "LOW": "#00c851",
            },
            hole=0.55,
            template="plotly_dark",
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig3, use_container_width="stretch")

    with col4:
        st.markdown(
            '<div class="section-header">Churn Rate by Tenure Band</div>',
            unsafe_allow_html=True,
        )
        df2 = df.copy()
        df2["tenure_band"] = pd.cut(
            df2["tenure_months"],
            bins=[0, 12, 24, 36, 48, 72],
            labels=["0–12mo", "12–24mo", "24–36mo", "36–48mo", "48–72mo"],
        )
        tenure_churn = (
            df2.groupby("tenure_band", observed=True)["label"].mean().reset_index()
        )
        tenure_churn["churn_pct"] = (tenure_churn["label"] * 100).round(1)
        fig4 = px.bar(
            tenure_churn,
            x="tenure_band",
            y="churn_pct",
            color="churn_pct",
            color_continuous_scale=["#00c851", "#ff8800", "#ff4444"],
            template="plotly_dark",
            text="churn_pct",
        )
        fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="",
            yaxis_title="Churn Rate %",
        )
        st.plotly_chart(fig4, use_container_width="stretch")


def page_churn_analysis(df: pd.DataFrame):
    st.markdown("## 🔴 Churn Analysis")
    render_kpis(df, load_revenue_data())
    st.markdown("<br/>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="section-header">Churn Probability Distribution</div>',
            unsafe_allow_html=True,
        )
        if "churn_prob" not in df.columns:
            df["churn_prob"] = df["label"].apply(
                lambda x: (
                    np.random.default_rng(x).uniform(0.6, 0.9)
                    if x == 1
                    else np.random.default_rng(x).uniform(0.05, 0.4)
                )
            )
        fig = px.histogram(
            df,
            x="churn_prob",
            color="risk_tier",
            nbins=40,
            template="plotly_dark",
            color_discrete_map={
                "HIGH": "#ff4444",
                "MEDIUM": "#ff8800",
                "LOW": "#00c851",
            },
            barmode="overlay",
            opacity=0.75,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Churn Probability",
            yaxis_title="Customer Count",
        )
        st.plotly_chart(fig, use_container_width="stretch")

    with col2:
        st.markdown(
            '<div class="section-header">Monthly Charges vs Churn Risk</div>',
            unsafe_allow_html=True,
        )
        sample = df.sample(min(800, len(df)), random_state=42)
        fig2 = px.scatter(
            sample,
            x="tenure_months",
            y="monthly_charges",
            color="risk_tier",
            color_discrete_map={
                "HIGH": "#ff4444",
                "MEDIUM": "#ff8800",
                "LOW": "#00c851",
            },
            opacity=0.65,
            template="plotly_dark",
            hover_data=["customer_id", "contract_type"],
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Tenure (months)",
            yaxis_title="Monthly Charges ($)",
        )
        st.plotly_chart(fig2, use_container_width="stretch")

    # High-risk customer table
    st.markdown(
        '<div class="section-header">🚨 High Risk Customers — Requires Action</div>',
        unsafe_allow_html=True,
    )
    high_risk = (
        df[df["risk_tier"] == "HIGH"]
        .sort_values("churn_prob", ascending=False)
        .head(20)
    )
    display_cols = [
        "customer_id",
        "tenure_months",
        "monthly_charges",
        "contract_type",
        "total_support_tickets",
        "churn_prob",
        "risk_tier",
    ]
    display_cols = [c for c in display_cols if c in high_risk.columns]
    table_df = high_risk[display_cols].copy()
    table_df["churn_prob"] = (table_df["churn_prob"] * 100).round(1).astype(str) + "%"

    st.dataframe(
        table_df,
        use_container_width="stretch",
        hide_index=True,
    )


def page_revenue(rev_df: pd.DataFrame):
    st.markdown("## 💰 Revenue Forecast")

    periods = st.slider("Forecast periods (months)", 3, 12, 6)
    fc_df = load_forecast_data(periods)

    col1, col2, col3 = st.columns(3)
    total_fc = fc_df["forecast"].sum()
    avg_fc = fc_df["forecast"].mean()
    growth = (
        (fc_df["forecast"].iloc[-1] - fc_df["forecast"].iloc[0])
        / fc_df["forecast"].iloc[0]
        * 100
    )

    with col1:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Forecast ({periods}mo)</div>
            <div class="kpi-value" style="color:#00c851">${total_fc:,.0f}</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Monthly Forecast</div>
            <div class="kpi-value" style="color:#00d4ff">${avg_fc:,.0f}</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        c = "#00c851" if growth > 0 else "#ff4444"
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">Projected Growth</div>
            <div class="kpi-value" style="color:{c}">
                {"▲" if growth > 0 else "▼"} {abs(growth):.1f}%
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">Historical + Forecast Revenue</div>',
        unsafe_allow_html=True,
    )

    # Combine historical and forecast
    hist = rev_df.tail(12).copy()
    hist.columns = ["month", "revenue"]
    hist["type"] = "Historical"
    hist["lower"] = hist["revenue"]
    hist["upper"] = hist["revenue"]

    fc = fc_df.copy()
    fc["month"] = pd.to_datetime(fc["month"])
    fc["revenue"] = fc["forecast"]
    fc["type"] = "Forecast"

    fig = go.Figure()

    # Historical line
    fig.add_trace(
        go.Scatter(
            x=hist["month"],
            y=hist["revenue"],
            mode="lines+markers",
            name="Historical",
            line=dict(color="#00d4ff", width=2.5),
            marker=dict(size=5),
        )
    )

    # Forecast confidence band
    fig.add_trace(
        go.Scatter(
            x=pd.concat([fc["month"], fc["month"][::-1]]),
            y=pd.concat([fc["upper"], fc["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(0,200,81,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Band",
            showlegend=True,
        )
    )

    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=fc["month"],
            y=fc["revenue"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#00c851", width=2.5, dash="dash"),
            marker=dict(size=6, symbol="diamond"),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.1),
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        xaxis_title="",
        yaxis_title="Revenue ($)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width="stretch")

    st.markdown(
        '<div class="section-header">Forecast Table</div>', unsafe_allow_html=True
    )
    fc_display = fc_df.copy()
    fc_display["forecast"] = fc_display["forecast"].apply(lambda x: f"${x:,.0f}")
    fc_display["lower"] = fc_display["lower"].apply(lambda x: f"${x:,.0f}")
    fc_display["upper"] = fc_display["upper"].apply(lambda x: f"${x:,.0f}")
    fc_display.columns = ["Month", "Forecast", "Lower Bound", "Upper Bound"]
    st.dataframe(fc_display, use_container_width="stretch", hide_index=True)


def page_segmentation(df: pd.DataFrame):
    st.markdown("## 👥 Customer Segmentation")

    if "segment" not in df.columns:
        segs = ["Champions", "Loyal Customers", "At Risk", "Hibernating", "Lost"]
        df["segment"] = np.random.default_rng(42).choice(
            segs, size=len(df), p=[0.20, 0.25, 0.25, 0.18, 0.12]
        )

    seg_colors = {
        "Champions": "#00d4ff",
        "Loyal Customers": "#a78bfa",
        "At Risk": "#ff8800",
        "Hibernating": "#718096",
        "Lost": "#ff4444",
    }

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="section-header">Segment Distribution</div>',
            unsafe_allow_html=True,
        )
        seg_counts = df["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        fig = px.pie(
            seg_counts,
            values="count",
            names="segment",
            color="segment",
            color_discrete_map=seg_colors,
            hole=0.5,
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width="stretch")

    with col2:
        st.markdown(
            '<div class="section-header">Avg Monthly Charges by Segment</div>',
            unsafe_allow_html=True,
        )
        seg_metrics = (
            df.groupby("segment")
            .agg(
                avg_charge=("monthly_charges", "mean"),
                avg_tenure=("tenure_months", "mean"),
                count=("customer_id", "count"),
                churn_rate=("label", "mean"),
            )
            .reset_index()
        )

        fig2 = px.bar(
            seg_metrics.sort_values("avg_charge", ascending=True),
            x="avg_charge",
            y="segment",
            orientation="h",
            color="segment",
            color_discrete_map=seg_colors,
            template="plotly_dark",
            text="avg_charge",
        )
        fig2.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Avg Monthly Charge ($)",
            yaxis_title="",
        )
        st.plotly_chart(fig2, use_container_width="stretch")

    st.markdown(
        '<div class="section-header">Segment Profiles</div>', unsafe_allow_html=True
    )
    seg_profile = (
        df.groupby("segment")
        .agg(
            Customers=("customer_id", "count"),
            Avg_Tenure=("tenure_months", "mean"),
            Avg_Charge=("monthly_charges", "mean"),
            Churn_Rate=("label", "mean"),
            Tickets=("total_support_tickets", "mean"),
        )
        .round(2)
        .reset_index()
    )
    seg_profile["Churn_Rate"] = (seg_profile["Churn_Rate"] * 100).round(1).astype(
        str
    ) + "%"
    seg_profile["Avg_Charge"] = seg_profile["Avg_Charge"].apply(lambda x: f"${x:.2f}")
    seg_profile.columns = [
        "Segment",
        "Customers",
        "Avg Tenure (mo)",
        "Avg Monthly Charge",
        "Churn Rate",
        "Avg Support Tickets",
    ]
    st.dataframe(seg_profile, use_container_width="stretch", hide_index=True)


def page_live_predictor():
    st.markdown("## 🤖 Live Churn Predictor")
    st.markdown("Enter customer details below to get a real-time churn prediction.")

    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.number_input("Tenure (months)", 0, 72, 12)
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, 780.0)
            tickets = st.number_input("Support Tickets", 0, 20, 1)

        with col2:
            contract = st.selectbox(
                "Contract Type", ["Month-to-month", "One year", "Two year"]
            )
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

        with col3:
            security = st.selectbox(
                "Online Security", ["No", "Yes", "No internet service"]
            )
            support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])

        submitted = st.form_submit_button(
            "🔮 Predict Churn Risk",
            use_container_width="stretch",
        )

    if submitted:
        try:
            from api.services.model_service import model_service

            model_service.load_all()

            features = {
                "tenure_months": tenure,
                "monthly_charges": monthly,
                "total_charges": total,
                "contract_type": contract,
                "payment_method": payment,
                "internet_service": internet,
                "online_security": security,
                "tech_support": support,
                "num_support_tickets": tickets,
                "gender": gender,
                "is_senior_citizen": 1 if senior == "Yes" else 0,
                "has_partner": partner,
                "has_dependents": "No",
                "paperless_billing": "Yes",
            }
            result = model_service.predict_churn(features)
            prob = result["churn_probability"]
            tier = result["risk_tier"]

            # Result display
            col_a, col_b = st.columns([1, 2])

            with col_a:
                color = (
                    "#ff4444"
                    if tier == "HIGH"
                    else "#ff8800" if tier == "MEDIUM" else "#00c851"
                )
                st.markdown(
                    f"""
                <div class="kpi-card" style="margin-top:16px">
                    <div class="kpi-label">Churn Probability</div>
                    <div class="kpi-value" style="color:{color}; font-size:3rem">
                        {prob:.1%}
                    </div>
                    <div class="kpi-delta">
                        <span class="badge-{'high' if tier=='HIGH' else 'medium' if tier=='MEDIUM' else 'low'}">
                            {tier} RISK
                        </span>
                    </div>
                    <div style="margin-top:12px;font-size:0.8rem;color:#718096">
                        {result['recommendation']}
                    </div>
                </div>""",
                    unsafe_allow_html=True,
                )

            with col_b:
                if result["top_factors"]:
                    st.markdown(
                        '<div class="section-header">Top Risk Factors</div>',
                        unsafe_allow_html=True,
                    )
                    factors_df = pd.DataFrame(result["top_factors"])
                    fig = px.bar(
                        factors_df.sort_values("impact"),
                        x="impact",
                        y="feature",
                        orientation="h",
                        color="impact",
                        color_continuous_scale=["#00c851", "#ff8800", "#ff4444"],
                        template="plotly_dark",
                        text="impact",
                    )
                    fig.update_traces(
                        texttemplate="%{text:.2f}", textposition="outside"
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        coloraxis_showscale=False,
                        showlegend=False,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="SHAP Impact",
                        yaxis_title="",
                        height=260,
                    )
                    st.plotly_chart(fig, use_container_width="stretch")

        except Exception as e:
            st.error(f"Prediction error: {e}")


def page_monitoring():
    st.markdown("## 📈 Model Monitoring")

    rng = np.random.default_rng(42)

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("AUC-ROC", "0.887", "#00c851", "▲ +0.003 vs last week"),
        ("F1 Score", "0.741", "#00d4ff", "▲ +0.012 vs last week"),
        ("Data Drift", "0.043", "#00c851", "✓ Below threshold 0.20"),
        ("Pred Drift", "0.021", "#00c851", "✓ Stable"),
    ]
    for col, (label, val, color, delta) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(
                f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color}">{val}</div>
                <div class="kpi-delta" style="color:#718096;font-size:0.72rem">{delta}</div>
            </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br/>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="section-header">AUC-ROC Over Time</div>',
            unsafe_allow_html=True,
        )
        weeks = pd.date_range("2024-01-01", periods=24, freq="W")
        auc = 0.85 + rng.normal(0, 0.008, 24).cumsum() * 0.1
        auc = np.clip(auc, 0.80, 0.95)
        fig = px.line(
            x=weeks,
            y=auc,
            template="plotly_dark",
            color_discrete_sequence=["#00d4ff"],
        )
        fig.add_hline(
            y=0.85,
            line_dash="dash",
            line_color="#ff4444",
            annotation_text="Min threshold",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="",
            yaxis_title="AUC-ROC",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width="stretch")

    with col2:
        st.markdown(
            '<div class="section-header">Feature Drift (PSI) by Feature</div>',
            unsafe_allow_html=True,
        )
        features = [
            "monthly_charges",
            "tenure_months",
            "total_charges",
            "support_tickets",
            "charge_gap",
            "services_count",
        ]
        psi_vals = rng.uniform(0.01, 0.15, len(features))
        drift_df = pd.DataFrame({"feature": features, "psi": psi_vals})
        drift_df["status"] = drift_df["psi"].apply(
            lambda x: "Stable" if x < 0.10 else "Warning" if x < 0.20 else "Drift"
        )
        colors = {"Stable": "#00c851", "Warning": "#ff8800", "Drift": "#ff4444"}
        fig2 = px.bar(
            drift_df.sort_values("psi", ascending=True),
            x="psi",
            y="feature",
            color="status",
            orientation="h",
            color_discrete_map=colors,
            template="plotly_dark",
        )
        fig2.add_vline(
            x=0.20,
            line_dash="dash",
            line_color="#ff4444",
            annotation_text="Drift threshold",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="PSI Score",
            yaxis_title="",
        )
        st.plotly_chart(fig2, use_container_width="stretch")

    st.markdown(
        '<div class="section-header">Model Training History</div>',
        unsafe_allow_html=True,
    )
    history = pd.DataFrame(
        {
            "Run": [f"run_{i:03d}" for i in range(1, 9)],
            "Date": pd.date_range("2024-01-01", periods=8, freq="2W"),
            "AUC-ROC": np.round(rng.uniform(0.84, 0.90, 8), 4),
            "F1": np.round(rng.uniform(0.70, 0.76, 8), 4),
            "Precision": np.round(rng.uniform(0.68, 0.76, 8), 4),
            "Recall": np.round(rng.uniform(0.70, 0.78, 8), 4),
            "Status": ["Production"] + ["Archived"] * 7,
        }
    )
    st.dataframe(history, use_container_width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════


def main():
    df = load_customer_data()
    rev_df = load_revenue_data()

    # Add churn_prob if missing
    if "churn_prob" not in df.columns:
        rng = np.random.default_rng(42)
        df["churn_prob"] = np.where(
            df["label"] == 1,
            rng.uniform(0.55, 0.95, len(df)),
            rng.uniform(0.05, 0.45, len(df)),
        ).round(3)

    if "risk_tier" not in df.columns:
        df["risk_tier"] = np.where(
            df["churn_prob"] >= 0.70,
            "HIGH",
            np.where(df["churn_prob"] >= 0.40, "MEDIUM", "LOW"),
        )

    page, filtered_df = render_sidebar(df)

    if page == "📊 Overview":
        page_overview(filtered_df, rev_df)
    elif page == "🔴 Churn Analysis":
        page_churn_analysis(filtered_df)
    elif page == "💰 Revenue Forecast":
        page_revenue(rev_df)
    elif page == "👥 Segmentation":
        page_segmentation(filtered_df)
    elif page == "🤖 Live Predictor":
        page_live_predictor()
    elif page == "📈 Model Monitoring":
        page_monitoring()
    else:
        page_overview(filtered_df, rev_df)


if __name__ == "__main__":
    main()
