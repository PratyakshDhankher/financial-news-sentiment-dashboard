"""
app.py  –  Financial News Sentiment & Stock Signal Dashboard
------------------------------------------------------------
Sprint 4: Streamlit interactive dashboard.

Run with:
    streamlit run dashboard/app.py
"""

import json
import os
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paths (relative to project root, one level above /dashboard)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "financial_news.db")
CORRELATION_CSV = os.path.join(BASE_DIR, "data", "correlation_results.csv")
QUALITY_JSON = os.path.join(BASE_DIR, "data", "data_quality.json")

TICKERS: list[str] = ["RELIANCE.NS", "INFY.NS", "ICICIBC.NS"]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS  – dark premium theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* App background */
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,.4);
    }
    [data-testid="metric-container"] label { color: #8b949e !important; font-size:.85rem; }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #e6edf3 !important; font-size: 1.7rem; font-weight: 700;
    }

    /* Section headings */
    h1 { color: #e6edf3 !important; font-weight: 700; }
    h2, h3 { color: #c9d1d9 !important; font-weight: 600; }

    /* Dataframe */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Expander */
    .streamlit-expanderHeader {
        background: #1c2128 !important;
        color: #c9d1d9 !important;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loaders – cached for performance
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_news_db() -> pd.DataFrame:
    """Load all rows from the news table; return empty DataFrame on failure."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT ticker, title, publisher, publish_time,
                   sentiment_label, confidence_score, sentiment_numeric
            FROM   news
            WHERE  sentiment_label IS NOT NULL
            ORDER  BY publish_time DESC
            """,
            conn,
            parse_dates=["publish_time"],
        )
        conn.close()
        df["publish_time"] = pd.to_datetime(df["publish_time"], utc=True)
        df["date"] = df["publish_time"].dt.normalize()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_correlation_data() -> pd.DataFrame:
    """Load the pre-computed correlation CSV."""
    if not os.path.exists(CORRELATION_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(CORRELATION_CSV, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_quality_report() -> dict:
    """Load the data quality JSON report."""
    if not os.path.exists(QUALITY_JSON):
        return {}
    try:
        with open(QUALITY_JSON, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Signal helper
# ---------------------------------------------------------------------------

def determine_signal(sentiment_score: float) -> tuple[str, str]:
    """
    Map an average sentiment score to a trading signal with emoji.

    Parameters
    ----------
    sentiment_score : float

    Returns
    -------
    tuple[str, str]
        (signal_label, delta_string_for_st_metric)
    """
    if sentiment_score > 0.2:
        return "🟢 Bullish", "positive"
    if sentiment_score < -0.2:
        return "🔴 Bearish", "inverse"
    return "⚪ Neutral", "off"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def build_dual_axis_chart(
    corr_df: pd.DataFrame,
    news_df: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """
    Build a dual-axis Plotly figure:
      - Left Y  : Daily sentiment score (colour-coded bar chart)
      - Right Y : Stock daily price-change % (line chart)

    Parameters
    ----------
    corr_df : pd.DataFrame
        Output of load_correlation_data() filtered to ``ticker``.
    news_df : pd.DataFrame
        Output of load_news_db() filtered to ``ticker``.
    ticker : str

    Returns
    -------
    go.Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    ticker_corr = corr_df[corr_df["ticker"] == ticker].sort_values("date")
    ticker_news = (
        news_df[news_df["ticker"] == ticker]
        .groupby("date")["sentiment_numeric"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_numeric": "avg_sentiment"})
        .sort_values("date")
    )

    # ---- Sentiment bars ----
    if not ticker_news.empty:
        bar_colors = [
            "#2ea043" if s > 0 else ("#8b949e" if s == 0 else "#f85149")
            for s in ticker_news["avg_sentiment"]
        ]
        fig.add_trace(
            go.Bar(
                x=ticker_news["date"],
                y=ticker_news["avg_sentiment"],
                name="Daily Sentiment",
                marker_color=bar_colors,
                opacity=0.85,
            ),
            secondary_y=False,
        )

    # ---- Price change line ----
    if not ticker_corr.empty:
        fig.add_trace(
            go.Scatter(
                x=ticker_corr["date"],
                y=ticker_corr["price_change_pct"],
                name="Price Change %",
                mode="lines+markers",
                line=dict(color="#58a6ff", width=2),
                marker=dict(size=5),
            ),
            secondary_y=True,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#c9d1d9"),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
    )
    fig.update_yaxes(
        title_text="Avg Sentiment Score",
        secondary_y=False,
        gridcolor="#21262d",
        color="#c9d1d9",
    )
    fig.update_yaxes(
        title_text="Price Change (%)",
        secondary_y=True,
        gridcolor="#21262d",
        color="#58a6ff",
    )
    fig.update_xaxes(gridcolor="#21262d", color="#8b949e")
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def main() -> None:
    """Render the full Streamlit dashboard."""

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center;margin-bottom:0'>📊 Financial News Sentiment Dashboard</h1>"
        "<p style='text-align:center;color:#8b949e;margin-top:4px'>"
        "FinBERT · SQLite · Correlation Engine · Live Signals</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Load data ─────────────────────────────────────────────────────────
    news_df = load_news_db()
    corr_df = load_correlation_data()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/stock-share.png",
            width=64,
        )
        st.markdown("### 🎯 Ticker Selector")
        selected_ticker = st.selectbox(
            "Choose a ticker",
            options=TICKERS,
            index=0,
        )
        st.markdown("---")
        st.markdown("**Pipeline Modules**")
        st.markdown(
            "✅ `fetch_news.py`  \n"
            "✅ `sentiment_analysis.py`  \n"
            "✅ `correlation_engine.py`  \n"
            "✅ `dashboard/app.py`",
        )

    # ── Filter data to selected ticker ────────────────────────────────────
    ticker_news = news_df[news_df["ticker"] == selected_ticker] if not news_df.empty else pd.DataFrame()
    ticker_corr = corr_df[corr_df["ticker"] == selected_ticker] if not corr_df.empty else pd.DataFrame()

    # ── Top Metrics ────────────────────────────────────────────────────────
    st.markdown(f"### 📌 {selected_ticker} — Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    # Current sentiment: average of last 5 headlines
    if not ticker_news.empty:
        last5_sentiment = ticker_news.head(5)["sentiment_numeric"].mean()
        signal_label, signal_delta_type = determine_signal(last5_sentiment)
        with col1:
            st.metric(
                label="Current Sentiment (last 5)",
                value=f"{last5_sentiment:+.3f}",
            )
        with col2:
            st.metric(label="Signal", value=signal_label)
    else:
        with col1:
            st.metric(label="Current Sentiment", value="N/A")
        with col2:
            st.metric(label="Signal", value="N/A")

    # 30-Day Correlation
    if not ticker_corr.empty and "pearson_correlation" in corr_df.columns:
        corr_row = corr_df[corr_df["ticker"] == selected_ticker]
        corr_val = corr_row["pearson_correlation"].values[0] if not corr_row.empty else float("nan")
        avg_price = corr_row["avg_price_change"].values[0] if not corr_row.empty else float("nan")
        with col3:
            st.metric(
                label="30-Day Pearson Correlation",
                value=f"{corr_val:+.4f}" if not pd.isna(corr_val) else "N/A",
            )
        with col4:
            st.metric(
                label="Avg Daily Price Change",
                value=f"{avg_price:+.4f}%" if not pd.isna(avg_price) else "N/A",
            )
    else:
        with col3:
            st.metric(label="30-Day Correlation", value="N/A")
        with col4:
            st.metric(label="Avg Price Change", value="N/A")

    st.divider()

    # ── Dual-Axis Chart ────────────────────────────────────────────────────
    st.markdown("### 📈 Sentiment vs. Stock Price Movement")

    if not ticker_news.empty and not ticker_corr.empty:
        fig = build_dual_axis_chart(corr_df, news_df, selected_ticker)
        st.plotly_chart(fig, use_container_width=True)
    elif not ticker_news.empty:
        # Sentiment only (no price data yet)
        daily_sent = (
            ticker_news.groupby("date")["sentiment_numeric"]
            .mean()
            .reset_index()
        )
        bar_colors = [
            "#2ea043" if s > 0 else ("#8b949e" if s == 0 else "#f85149")
            for s in daily_sent["sentiment_numeric"]
        ]
        fig = go.Figure(
            go.Bar(
                x=daily_sent["date"],
                y=daily_sent["sentiment_numeric"],
                marker_color=bar_colors,
                name="Daily Sentiment",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No chart data available. Run the pipeline scripts first.")

    st.divider()

    # ── Raw Data Table ─────────────────────────────────────────────────────
    st.markdown("### 🗞️ Latest 10 Headlines")
    if not ticker_news.empty:
        display_cols = ["title", "publisher", "sentiment_label", "confidence_score"]
        latest = ticker_news[display_cols].head(10).rename(
            columns={
                "title": "Headline",
                "publisher": "Publisher",
                "sentiment_label": "Sentiment",
                "confidence_score": "Confidence",
            }
        )
        st.dataframe(
            latest,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(f"No headlines found for {selected_ticker}. Run fetch_news.py first.")

    st.divider()

    # ── Data Quality Report ────────────────────────────────────────────────
    with st.expander("🔍 Data Quality Report", expanded=False):
        quality = load_quality_report()
        if quality:
            q_col1, q_col2, q_col3 = st.columns(3)
            with q_col1:
                st.metric("Total Records", quality.get("total_records", "N/A"))
            with q_col2:
                st.metric("Duplicates Removed", quality.get("duplicate_count", "N/A"))
            with q_col3:
                clean_pct = quality.get("clean_data_percentage", "N/A")
                st.metric(
                    "Clean Data %",
                    f"{clean_pct}%" if isinstance(clean_pct, (int, float)) else "N/A",
                )
            st.caption(f"Report generated at: {quality.get('generated_at', 'unknown')}")
        else:
            st.warning(
                "data_quality.json not found. "
                "Run `python scripts/fetch_news.py` to generate it."
            )

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown(
        "<hr style='border-color:#30363d'>"
        "<p style='text-align:center;color:#8b949e;font-size:.8rem'>"
        "Financial Sentiment Dashboard · Powered by FinBERT · Built with Streamlit & Plotly"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
