"""
fetch_news.py
-------------
Sprint 1 (v2): Financial News Ingestion & Validation Pipeline

News sources:
  1. Yahoo Finance  – via yfinance (existing)
  2. NewsAPI        – via requests  (new, requires NEWSAPI_KEY env var)
  3. Google News RSS – via feedparser (new)

Target volume: 500–1 000 deduplicated records per run.

Usage:
    python scripts/fetch_news.py
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import feedparser
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR  = os.path.join(BASE_DIR, "logs")
DB_PATH              = os.path.join(DATA_DIR, "financial_news.db")
QUALITY_REPORT_PATH  = os.path.join(DATA_DIR, "data_quality.json")
LOG_PATH             = os.path.join(LOG_DIR,  "pipeline.log")

# Target tickers
TICKERS: list[str] = ["RELIANCE.NS", "INFY.NS", "ICICIBC.NS"]

# NewsAPI key – set the NEWSAPI_KEY environment variable before running
NEWSAPI_KEY: str = os.environ.get("NEWSAPI_KEY", "")

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

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
# Database helpers  (unchanged)
# ---------------------------------------------------------------------------

def get_db_connection() -> sqlite3.Connection:
    """
    Return a SQLite connection object for consistent database access.

    Returns
    -------
    sqlite3.Connection
        An open connection to the financial_news.db database.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    """
    Create the SQLite database and the `news` table if they do not exist.

    The table schema includes placeholder columns for sentiment fields that
    will be populated by the downstream sentiment_analysis.py script.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS news (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker              TEXT,
            title               TEXT,
            publisher           TEXT,
            link                TEXT,
            publish_time        DATETIME,
            sentiment_label     TEXT,
            confidence_score    REAL,
            sentiment_numeric   INTEGER
        )
        """
    )
    conn.commit()
    conn.close()
    logger.info("Database initialized at: %s", DB_PATH)


# ---------------------------------------------------------------------------
# Source 1 – Yahoo Finance  (existing, unchanged)
# ---------------------------------------------------------------------------

def fetch_news_for_ticker(ticker_symbol: str) -> list[dict]:
    """
    Fetch raw news articles for a single ticker using yfinance.

    Parameters
    ----------
    ticker_symbol : str
        The Yahoo Finance ticker symbol (e.g., "RELIANCE.NS").

    Returns
    -------
    list[dict]
        A list of normalized news dictionaries ready for DB insertion.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        raw_news: list[dict] = ticker.news
        records: list[dict] = []

        for item in raw_news:
            content: dict = item.get("content", item)

            # Extract timestamp
            raw_ts: Optional[int] = (
                item.get("providerPublishTime")
                or content.get("providerPublishTime")
            )

            # --- Robust timestamp handling ---
            if raw_ts is not None:
                publish_time = pd.to_datetime(raw_ts, unit="s", utc=True)
            else:
                logger.warning(
                    "Missing publish time for article: %s",
                    content.get("title") or item.get("title"),
                )
                publish_time = pd.Timestamp.utcnow()

            publish_time = str(publish_time)

            records.append(
                {
                    "ticker": ticker_symbol,
                    "title": content.get("title") or item.get("title"),
                    "publisher": (
                        content.get("provider", {}).get("displayName")
                        if isinstance(content.get("provider"), dict)
                        else item.get("publisher")
                    ),
                    "link": (
                        content.get("canonicalUrl", {}).get("url")
                        if isinstance(content.get("canonicalUrl"), dict)
                        else item.get("link")
                    ),
                    "publish_time": publish_time,
                }
            )

        logger.info(
            "[Yahoo Finance] Fetched %d articles for ticker: %s",
            len(records),
            ticker_symbol,
        )

        return records

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Yahoo Finance API failure for ticker %s: %s",
            ticker_symbol,
            exc,
            exc_info=True,
        )
        return []


def fetch_all_yahoo_news(tickers: list[str]) -> list[dict]:
    """
    Fetch and combine Yahoo Finance news for all target tickers.

    Parameters
    ----------
    tickers : list[str]
        List of Yahoo Finance ticker symbols.

    Returns
    -------
    list[dict]
        Combined list of news record dicts.
    """
    all_records: list[dict] = []
    for symbol in tickers:
        all_records.extend(fetch_news_for_ticker(symbol))

    logger.info(
        "[Yahoo Finance] Total articles fetched across all tickers: %d",
        len(all_records),
    )
    return all_records


# ---------------------------------------------------------------------------
# Source 2 – NewsAPI
# ---------------------------------------------------------------------------

def fetch_news_from_newsapi(query: str, pages: int = 10) -> list[dict]:
    """
    Fetch financial news articles from NewsAPI.

    Retrieves up to 100 articles per page and supports pagination so that
    large volumes of headlines can be collected in a single run.

    Parameters
    ----------
    query : str
        Free-text search query (e.g., "Reliance Industries stock").
    pages : int, optional
        Number of result pages to fetch (default 10 → up to 1 000 articles).

    Returns
    -------
    list[dict]
        Normalized news records conforming to the project schema.

    Notes
    -----
    Requires the ``NEWSAPI_KEY`` environment variable to be set.
    Free-plan accounts are limited to page 1; a paid key is needed for
    pagination beyond the first page.
    """
    if not NEWSAPI_KEY:
        logger.warning(
            "[NewsAPI] NEWSAPI_KEY env var not set – skipping NewsAPI source."
        )
        return []

    base_url = "https://newsapi.org/v2/everything"
    records: list[dict] = []

    for page in range(1, pages + 1):
        params = {
            "q":        query,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": 100,
            "page":     page,
            "apiKey":   NEWSAPI_KEY,
        }
        try:
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()

            articles = payload.get("articles", [])
            if not articles:
                logger.info(
                    "[NewsAPI] No more articles at page %d – stopping.", page
                )
                break

            for article in articles:
                # Convert publishedAt ISO-8601 string → UTC timestamp string
                raw_ts: str = article.get("publishedAt", "")
                try:
                    publish_time = str(
                        pd.to_datetime(raw_ts, utc=True)
                    )
                except Exception:
                    publish_time = str(pd.Timestamp.utcnow())

                records.append(
                    {
                        "ticker":       query,          # best proxy for ticker context
                        "title":        article.get("title") or "",
                        "publisher":    (article.get("source") or {}).get("name") or "",
                        "link":         article.get("url") or "",
                        "publish_time": publish_time,
                    }
                )

            logger.info(
                "[NewsAPI] Page %d → fetched %d articles (running total: %d)",
                page,
                len(articles),
                len(records),
            )

            # NewsAPI caps free results at 100 total; break early if exhausted
            if len(articles) < 100:
                break

        except requests.RequestException as exc:
            logger.error(
                "[NewsAPI] Request failed at page %d: %s", page, exc, exc_info=True
            )
            break

    logger.info(
        "[NewsAPI] Total articles fetched for query '%s': %d", query, len(records)
    )
    return records


# ---------------------------------------------------------------------------
# Source 3 – Google News RSS
# ---------------------------------------------------------------------------

def fetch_news_from_google_rss(query: str) -> list[dict]:
    """
    Fetch financial news headlines from the Google News RSS feed.

    Parameters
    ----------
    query : str
        Search query appended to the Google News RSS URL.

    Returns
    -------
    list[dict]
        Normalized news records conforming to the project schema.

    Notes
    -----
    Uses ``feedparser`` to parse the Atom/RSS feed returned by Google News.
    The number of results depends on Google's feed cap (~100 entries typical).
    """
    feed_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    records: list[dict] = []

    try:
        feed = feedparser.parse(feed_url)

        if feed.bozo and not feed.entries:
            logger.warning(
                "[Google RSS] Feed parse error for query '%s': %s",
                query,
                feed.bozo_exception,
            )
            return []

        for entry in feed.entries:
            # Attempt to extract publisher from <source> tag
            publisher: str = ""
            if hasattr(entry, "source") and isinstance(entry.source, dict):
                publisher = entry.source.get("title", "")
            elif hasattr(entry, "tags") and entry.tags:
                publisher = entry.tags[0].get("term", "")

            # Parse publish time
            raw_published: str = getattr(entry, "published", "")
            try:
                dt = parsedate_to_datetime(raw_published)
                publish_time = str(
                    pd.Timestamp(dt).tz_convert("UTC")
                    if dt.tzinfo
                    else pd.Timestamp(dt, tz="UTC")
                )
            except Exception:
                publish_time = str(pd.Timestamp.utcnow())

            records.append(
                {
                    "ticker":       query,
                    "title":        getattr(entry, "title", "") or "",
                    "publisher":    publisher,
                    "link":         getattr(entry, "link", "") or "",
                    "publish_time": publish_time,
                }
            )

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[Google RSS] Failed to fetch feed for query '%s': %s",
            query,
            exc,
            exc_info=True,
        )

    logger.info(
        "[Google RSS] Fetched %d articles for query '%s'", len(records), query
    )
    return records


# ---------------------------------------------------------------------------
# Aggregator – combine all three sources
# ---------------------------------------------------------------------------

def fetch_all_news(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch and combine news from Yahoo Finance, NewsAPI, and Google RSS into
    a single DataFrame.

    For NewsAPI and Google RSS the ticker symbol is used as the search query
    (after stripping the exchange suffix, e.g. "RELIANCE" from "RELIANCE.NS").

    Parameters
    ----------
    tickers : list[str]
        List of Yahoo Finance ticker symbols to fetch news for.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns:
        ticker, title, publisher, link, publish_time.
    """
    all_records: list[dict] = []

    # --- Source 1: Yahoo Finance ---
    yahoo_records = fetch_all_yahoo_news(tickers)
    all_records.extend(yahoo_records)
    logger.info("[Aggregator] Yahoo Finance contributed %d records.", len(yahoo_records))

    # --- Sources 2 & 3: NewsAPI + Google RSS (one query per ticker) ---
    for ticker_symbol in tickers:
        # Use the root ticker name as search query (drop exchange suffix)
        query = ticker_symbol.split(".")[0]

        newsapi_records = fetch_news_from_newsapi(query, pages=10)
        all_records.extend(newsapi_records)
        logger.info(
            "[Aggregator] NewsAPI contributed %d records for '%s'.",
            len(newsapi_records),
            query,
        )

        rss_records = fetch_news_from_google_rss(query)
        all_records.extend(rss_records)
        logger.info(
            "[Aggregator] Google RSS contributed %d records for '%s'.",
            len(rss_records),
            query,
        )

    if not all_records:
        logger.warning("No news records fetched from any source.")
        return pd.DataFrame(
            columns=["ticker", "title", "publisher", "link", "publish_time"]
        )

    df = pd.DataFrame(all_records)

    # --- Deduplication before returning ---
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["title"], keep="first")
    dupes_removed = before_dedup - len(df)
    logger.info(
        "[Aggregator] Deduplication: %d duplicates removed → %d unique records remain.",
        dupes_removed,
        len(df),
    )

    # --- Normalize publish_time to UTC string ---
    def _normalize_ts(val: object) -> str:
        try:
            ts = pd.to_datetime(val, utc=True)
            return str(ts)
        except Exception:
            return str(pd.Timestamp.utcnow())

    df["publish_time"] = df["publish_time"].apply(_normalize_ts)

    logger.info("[Aggregator] Final combined record count: %d", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation & quality reporting  (unchanged)
# ---------------------------------------------------------------------------

def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw news DataFrame and write a data-quality report to disk.

    Steps performed:
        1. Count and drop duplicate headlines (based on ``title``).
        2. Remove rows where ``title`` or ``publisher`` are missing.
        3. Compute quality metrics and persist them as JSON.

    Parameters
    ----------
    df : pd.DataFrame
        Raw news DataFrame produced by :func:`fetch_all_news`.

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame ready for database insertion.
    """
    total_records: int = len(df)

    # --- Deduplication (safety pass – fetch_all_news already deduped) ---
    duplicate_count: int = df.duplicated(subset=["title"], keep="first").sum()
    df = df.drop_duplicates(subset=["title"], keep="first")

    # --- Drop rows with missing critical fields ---
    df = df.dropna(subset=["title", "publisher"])
    df = df[df["title"].str.strip().ne("") & df["publisher"].str.strip().ne("")]

    clean_records: int = len(df)
    clean_data_percentage: float = (
        round((clean_records / total_records) * 100, 2) if total_records else 0.0
    )

    quality_report: dict = {
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "total_records":       total_records,
        "duplicate_count":     int(duplicate_count),
        "clean_records":       clean_records,
        "clean_data_percentage": clean_data_percentage,
    }

    with open(QUALITY_REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(quality_report, fh, indent=4)

    logger.info(
        "Data quality report saved → total=%d | duplicates=%d | clean=%.1f%%",
        total_records,
        duplicate_count,
        clean_data_percentage,
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Database insertion  (unchanged)
# ---------------------------------------------------------------------------

def insert_records(df: pd.DataFrame) -> None:
    """
    Insert validated news records into the ``news`` SQLite table.

    Only inserts rows whose ``title`` does not already exist in the database
    to prevent duplicate entries across multiple runs.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame to persist.
    """
    if df.empty:
        logger.warning("No records to insert – DataFrame is empty.")
        return

    conn = get_db_connection()
    cursor = conn.cursor()

    inserted: int = 0
    skipped: int  = 0

    for _, row in df.iterrows():
        try:
            # Skip if headline already stored
            cursor.execute(
                "SELECT id FROM news WHERE title = ?", (row["title"],)
            )
            if cursor.fetchone():
                skipped += 1
                continue

            cursor.execute(
                """
                INSERT INTO news
                    (ticker, title, publisher, link, publish_time)
                VALUES
                    (?, ?, ?, ?, ?)
                """,
                (
                    row.get("ticker"),
                    row.get("title"),
                    row.get("publisher"),
                    row.get("link"),
                    row.get("publish_time"),
                ),
            )
            inserted += 1

        except sqlite3.Error as db_err:
            logger.error("DB insert error for title '%s': %s", row.get("title"), db_err)

    conn.commit()
    conn.close()
    logger.info(
        "Database insert complete → inserted=%d | skipped(duplicate)=%d",
        inserted,
        skipped,
    )
    logger.info(
        "[Summary] Final record count inserted into DB: %d", inserted
    )


# ---------------------------------------------------------------------------
# Entry point  (unchanged)
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrates the full news ingestion pipeline."""
    logger.info("=" * 60)
    logger.info("Financial News Ingestion Pipeline – START")
    logger.info("=" * 60)

    # Step 1 – Initialize DB
    initialize_database()

    # Step 2 – Fetch news from all three sources
    raw_df: pd.DataFrame = fetch_all_news(TICKERS)
    logger.info("Total raw articles fetched (post-dedup): %d", len(raw_df))

    if raw_df.empty:
        logger.warning("Pipeline terminated early – no data available.")
        return

    # Step 3 – Clean & validate
    clean_df: pd.DataFrame = clean_and_validate(raw_df)
    logger.info("Articles after cleaning: %d", len(clean_df))

    # Step 4 – Persist to database
    insert_records(clean_df)

    logger.info("Financial News Ingestion Pipeline – COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
