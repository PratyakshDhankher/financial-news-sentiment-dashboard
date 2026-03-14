"""
correlation_engine.py
----------------------
Sprint 3: Sentiment–Stock Price Correlation Analysis

Fetches 30-day closing prices for target tickers from yfinance, merges them
with daily average sentiment scores stored in the SQLite database, and
computes Pearson correlation coefficients.  Results are printed as a summary
table and saved to data/correlation_results.csv for use by the dashboard.

Usage:
    python scripts/correlation_engine.py
"""

import logging
import os
import sqlite3
from typing import Optional

import pandas as pd
import yfinance as yf
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "financial_news.db")
LOG_PATH = os.path.join(BASE_DIR, "logs", "pipeline.log")
OUTPUT_CSV = os.path.join(DATA_DIR, "correlation_results.csv")

TICKERS = ["RELIANCE.NS", "INFY.NS", "ICICIBANK.NS"]
LOOKBACK_DAYS: int = 30

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db_connection() -> sqlite3.Connection:
    """
    Return a SQLite connection for consistent database access.

    Returns
    -------
    sqlite3.Connection
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_sentiment_from_db() -> pd.DataFrame:
    """
    Load all rows from the ``news`` table that have a numeric sentiment score.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, publish_time (DatetimeTZDtype UTC), sentiment_numeric.
    """
    conn = get_db_connection()
    df = pd.read_sql_query(
        """
        SELECT ticker, publish_time, sentiment_numeric
        FROM   news
        WHERE  sentiment_numeric IS NOT NULL
        """,
        conn,
        parse_dates=["publish_time"],
    )
    conn.close()

    if df.empty:
        logger.warning("No sentiment data found in the database.")
        return df

    # Normalise timezone-aware datetime
    df["publish_time"] = pd.to_datetime(df["publish_time"], utc=True)
    logger.info("Loaded %d sentiment records from database.", len(df))
    return df


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def fetch_price_data(tickers: list[str], period_days: int = 30) -> pd.DataFrame:
    """
    Fetch daily closing prices for the given tickers over the last N days.

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance ticker symbols.
    period_days : int
        Number of calendar days of history to retrieve.

    Returns
    -------
    pd.DataFrame
        Multi-column DataFrame with tickers as column names and
        UTC-localised DatetimeIndex.
    """
    period_str = f"{period_days}d"
    try:
        raw = yf.download(
            tickers,
            period=period_str,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            logger.error("yfinance returned no price data.")
            return pd.DataFrame()

        # Extract 'Close' prices
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            prices.columns = [tickers[0]]

        prices.index = pd.to_datetime(prices.index, utc=True)
        logger.info(
            "Fetched %d days of price data for %d tickers.",
            len(prices),
            len(prices.columns),
        )
        return prices

    except Exception as exc:  # noqa: BLE001
        logger.error("Price data fetch failed: %s", exc, exc_info=True)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Aggregation & merge
# ---------------------------------------------------------------------------

def compute_daily_sentiment(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample sentiment records with robust handling for null timestamps.
    """
    # 1. Convert to datetime
    sentiment_df["publish_time"] = pd.to_datetime(sentiment_df["publish_time"], utc=True)

    # 2. DROP NULL TIMESTAMPS (The fix for your NaT error)
    # This removes any row where the date couldn't be parsed
    sentiment_df = sentiment_df.dropna(subset=["publish_time"])

    # 3. Safety check: If no data left, return an empty DF with correct columns
    if sentiment_df.empty:
        logger.warning("No valid timestamps found after dropping NaT.")
        return pd.DataFrame()

    # 4. Group by ticker and day
    daily = (
        sentiment_df.groupby([
            "ticker", 
            pd.Grouper(key="publish_time", freq="D")
        ])["sentiment_numeric"]
        .mean()
        .reset_index()
    )

    # 5. Normalize (Safety check: only normalize if not NaT)
    daily["publish_time"] = daily["publish_time"].dt.normalize()

    # 6. Pivot
    pivoted = daily.pivot_table(
        index="publish_time",
        columns="ticker",
        values="sentiment_numeric"
    )

    # Ensure index is datetime for the price merge
    pivoted.index = pd.to_datetime(pivoted.index, utc=True)

    logger.info(f"Daily sentiment aggregated: {len(pivoted)} rows.")
    return pivoted


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentage daily price change for each ticker.

    Parameters
    ----------
    prices : pd.DataFrame
        Closing price DataFrame with ticker columns.

    Returns
    -------
    pd.DataFrame
        Percentage change DataFrame (NaN on first row dropped).
    """
    returns = prices.pct_change() * 100
    returns = returns.dropna(how="all")
    returns.index = returns.index.normalize()
    return returns


def merge_datasets(
    daily_sentiment: pd.DataFrame,
    daily_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join daily sentiment averages with daily price returns on the date index.

    Parameters
    ----------
    daily_sentiment : pd.DataFrame
        Pivoted daily sentiment (index = UTC date).
    daily_returns : pd.DataFrame
        Daily percentage price changes (index = UTC date).

    Returns
    -------
    pd.DataFrame
        Merged long-format DataFrame with columns:
        date, ticker, avg_sentiment, price_change_pct.
    """
    rows: list[dict] = []

    for ticker in TICKERS:
        if ticker not in daily_sentiment.columns or ticker not in daily_returns.columns:
            logger.warning("Ticker %s missing from one of the datasets; skipping.", ticker)
            continue

        sent_series = daily_sentiment[ticker].dropna()
        ret_series = daily_returns[ticker].dropna()

        merged = pd.merge(
            sent_series.rename("avg_sentiment"),
            ret_series.rename("price_change_pct"),
            left_index=True,
            right_index=True,
            how="outer",
        )

        merged["avg_sentiment"] = merged["avg_sentiment"].ffill()
        merged = merged.dropna(subset=["price_change_pct"])

        merged["ticker"] = ticker
        merged = merged.reset_index().rename(columns={"publish_time": "date"})
        rows.append(merged)

    if not rows:
        logger.error("No overlapping data after merge. Check date ranges.")
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    logger.info("Merged dataset: %d rows.", len(combined))
    return combined

# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_correlation(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Pearson correlation between sentiment and price change per ticker.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Output of :func:`merge_datasets`.

    Returns
    -------
    pd.DataFrame
        Summary table with columns:
        ticker, avg_sentiment, avg_price_change, pearson_correlation, p_value.
    """
    summary_rows: list[dict] = []

    for ticker in merged_df["ticker"].unique():
        sub = merged_df[merged_df["ticker"] == ticker]

        avg_sentiment: float = sub["avg_sentiment"].mean()
        avg_price_change: float = sub["price_change_pct"].mean()

        pearson_r: float = float("nan")
        p_value: float = float("nan")

        if len(sub) >= 2:
            try:
                result = stats.pearsonr(sub["avg_sentiment"], sub["price_change_pct"])
                pearson_r = result.statistic
                p_value = result.pvalue
            except Exception as exc:  # noqa: BLE001
                logger.warning("Pearson calculation failed for %s: %s", ticker, exc)

        summary_rows.append(
            {
                "ticker": ticker,
                "avg_sentiment": round(avg_sentiment, 4),
                "avg_price_change": round(avg_price_change, 4),
                "pearson_correlation": round(pearson_r, 4),
                "p_value": round(p_value, 4),
            }
        )
        logger.info(
            "%-12s | avg_sentiment=%+.4f | avg_price_change=%+.4f%% | r=%+.4f | p=%.4f",
            ticker,
            avg_sentiment,
            avg_price_change,
            pearson_r,
            p_value,
        )

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrates the full correlation analysis pipeline."""
    logger.info("=" * 60)
    logger.info("Correlation Engine – START")
    logger.info("=" * 60)

    # Step 1 – Load sentiment from DB
    sentiment_df = load_sentiment_from_db()
    if sentiment_df.empty:
        logger.warning("No sentiment data – run sentiment_analysis.py first.")
        return

    # Step 2 – Fetch 30-day price data
    prices = fetch_price_data(TICKERS, period_days=LOOKBACK_DAYS)
    if prices.empty:
        logger.error("No price data fetched. Aborting.")
        return

    # Step 3 – Compute daily aggregates
    daily_sentiment = compute_daily_sentiment(sentiment_df)
    daily_returns = compute_daily_returns(prices)

    # Step 4 – Merge
    merged_df = merge_datasets(daily_sentiment, daily_returns)
    if merged_df.empty:
        logger.error("Merged dataset is empty. Cannot compute correlations.")
        return

    # Step 5 – Correlation
    summary = compute_correlation(merged_df)

    # Step 6 – Print summary table
    print("\n" + "=" * 64)
    print(" CORRELATION SUMMARY")
    print("=" * 64)
    print(summary.to_string(index=False))
    print("=" * 64 + "\n")

    # Step 7 – Persist merged dataset for the dashboard
    # Ensure date column is serialisable
    merged_df["date"] = merged_df["date"].astype(str)
    merged_df.to_csv(OUTPUT_CSV, index=False)
    logger.info("Correlation results saved → %s", OUTPUT_CSV)

    logger.info("Correlation Engine – COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
