"""
sentiment_analysis.py
----------------------
Sprint 2: FinBERT Sentiment Analysis on Stored News Headlines

This script loads unanalysed news records from the SQLite database,
runs them through ProsusAI/finbert (with GPU support when available),
and writes back the sentiment label, confidence score, and numeric
encoding to each row.

Usage:
    python scripts/sentiment_analysis.py
"""

import logging
import os
import sqlite3
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "financial_news.db")
LOG_PATH = os.path.join(BASE_DIR, "logs", "pipeline.log")

# FinBERT model identifier on HuggingFace Hub
MODEL_NAME: str = "ProsusAI/finbert"

# Batch size for inference
BATCH_SIZE: int = 16

# Label → numeric mapping
SENTIMENT_MAP: dict[str, int] = {
    "positive": 1,
    "neutral": 0,
    "negative": -1,
}

# ---------------------------------------------------------------------------
# Logging configuration (append to the same log created by fetch_news.py)
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
    Return a SQLite connection object for consistent database access.

    Returns
    -------
    sqlite3.Connection
        An open connection to the financial_news.db database.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_unanalysed_records() -> list[dict]:
    """
    Query the news table for rows where ``sentiment_label`` is NULL.

    Returns
    -------
    list[dict]
        A list of row dicts with keys ``id`` and ``title``.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title FROM news WHERE sentiment_label IS NULL"
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    logger.info("Found %d unanalysed records.", len(rows))
    return rows


def update_sentiment(
    record_id: int,
    sentiment_label: str,
    confidence_score: float,
    sentiment_numeric: int,
) -> None:
    """
    Write sentiment results back to the database for a single record.

    Parameters
    ----------
    record_id : int
        Primary key of the news row to update.
    sentiment_label : str
        One of ``"positive"``, ``"neutral"``, or ``"negative"``.
    confidence_score : float
        Probability of the predicted class (0.0 – 1.0).
    sentiment_numeric : int
        Integer encoding: +1, 0, or -1.
    """
    conn = get_db_connection()
    conn.execute(
        """
        UPDATE news
        SET sentiment_label   = ?,
            confidence_score  = ?,
            sentiment_numeric = ?
        WHERE id = ?
        """,
        (sentiment_label, confidence_score, sentiment_numeric, record_id),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_name: str,
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    """
    Load the FinBERT tokenizer and model, moving to GPU if available.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g., ``"ProsusAI/finbert"``).

    Returns
    -------
    tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]
        The tokenizer, model, and the device the model was placed on.

    Raises
    ------
    RuntimeError
        If the model cannot be loaded (e.g., no internet, corrupt cache).
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        logger.info("Loading tokenizer from: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info("Loading model from: %s", model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()

        logger.info("Model loaded successfully and set to eval mode.")
        return tokenizer, model, device

    except Exception as exc:
        logger.error("Model loading failed: %s", exc, exc_info=True)
        raise RuntimeError(f"Failed to load model '{model_name}': {exc}") from exc


# ---------------------------------------------------------------------------
# Sentiment inference
# ---------------------------------------------------------------------------

def analyze_sentiment(
    headlines: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
) -> list[dict]:
    """
    Run FinBERT sentiment inference on a list of headlines in batches.

    FinBERT's label ordering is: positive (0), negative (1), neutral (2).
    Softmax is applied to the raw logits to obtain per-class probabilities.
    The predicted class is the one with the highest probability.

    Parameters
    ----------
    headlines : list[str]
        List of news headline strings to analyse.
    tokenizer : AutoTokenizer
        Pre-loaded FinBERT tokenizer.
    model : AutoModelForSequenceClassification
        Pre-loaded FinBERT model (already on ``device``).
    device : torch.device
        The torch device (CPU or CUDA) the model resides on.

    Returns
    -------
    list[dict]
        One dict per headline, each containing:
        ``sentiment_label``, ``confidence_score``, ``sentiment_numeric``.
    """
    # FinBERT's id2label mapping from its config
    id2label: dict[int, str] = {0: "positive", 1: "negative", 2: "neutral"}

    results: list[dict] = []

    # Process in batches for memory efficiency
    for batch_start in range(0, len(headlines), BATCH_SIZE):
        batch = headlines[batch_start : batch_start + BATCH_SIZE]

        try:
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            # Move tensors to the same device as the model
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Apply softmax to logits → probability distribution
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            for prob_row in probs:
                # Highest probability class
                confidence_score: float = prob_row.max().item()
                predicted_idx: int = prob_row.argmax().item()
                sentiment_label: str = id2label[predicted_idx]
                sentiment_numeric: int = SENTIMENT_MAP[sentiment_label]

                results.append(
                    {
                        "sentiment_label": sentiment_label,
                        "confidence_score": round(confidence_score, 4),
                        "sentiment_numeric": sentiment_numeric,
                    }
                )

        except torch.cuda.OutOfMemoryError:
            logger.error(
                "GPU out of memory on batch starting at index %d. "
                "Try reducing BATCH_SIZE.",
                batch_start,
                exc_info=True,
            )
            # Fall back: append neutral placeholders for the failed batch
            for _ in batch:
                results.append(
                    {
                        "sentiment_label": "neutral",
                        "confidence_score": 0.0,
                        "sentiment_numeric": 0,
                    }
                )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Inference error on batch at index %d: %s",
                batch_start,
                exc,
                exc_info=True,
            )
            for _ in batch:
                results.append(
                    {
                        "sentiment_label": "neutral",
                        "confidence_score": 0.0,
                        "sentiment_numeric": 0,
                    }
                )

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrates the full sentiment analysis pipeline."""
    logger.info("=" * 60)
    logger.info("Sentiment Analysis Pipeline – START")
    logger.info("=" * 60)

    # Step 1 – Fetch unanalysed records
    records = fetch_unanalysed_records()

    if not records:
        logger.info("All records already have sentiment labels. Nothing to do.")
        return

    # Step 2 – Load model
    try:
        tokenizer, model, device = load_model(MODEL_NAME)
    except RuntimeError:
        logger.critical("Cannot proceed without model. Exiting.")
        return

    # Step 3 – Run inference with tqdm progress tracking
    headlines: list[str] = [r["title"] for r in records]
    ids: list[int] = [r["id"] for r in records]

    logger.info(
        "Starting inference on %d headlines (batch_size=%d).",
        len(headlines),
        BATCH_SIZE,
    )

    sentiments: list[dict] = []

    # Wrap batch iterations with tqdm for a visible progress bar
    num_batches = (len(headlines) + BATCH_SIZE - 1) // BATCH_SIZE
    with tqdm(total=num_batches, desc="Analysing batches", unit="batch") as pbar:
        for batch_start in range(0, len(headlines), BATCH_SIZE):
            batch_headlines = headlines[batch_start : batch_start + BATCH_SIZE]
            batch_results = analyze_sentiment(
                batch_headlines, tokenizer, model, device
            )
            sentiments.extend(batch_results)
            pbar.update(1)

    # Step 4 – Persist results
    logger.info("Writing sentiment results to database …")
    for record_id, sentiment in zip(ids, sentiments):
        update_sentiment(
            record_id=record_id,
            sentiment_label=sentiment["sentiment_label"],
            confidence_score=sentiment["confidence_score"],
            sentiment_numeric=sentiment["sentiment_numeric"],
        )

    logger.info(
        "Sentiment analysis complete → %d records updated.", len(sentiments)
    )
    logger.info("Sentiment Analysis Pipeline – COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
