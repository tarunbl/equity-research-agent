"""
utils/finbert.py
================
Optional FinBERT financial sentiment classifier.

FinBERT (ProsusAI/finbert) is a BERT model fine-tuned on financial text.
It classifies text as positive, negative, or neutral with better accuracy
than general-purpose models on financial language ("headwinds", "guides lower",
"misses estimates" etc.).

Usage
-----
Install optional dependencies:
    pip install transformers torch

Then FinBERT is used automatically by the News Agent for headline
classification. Without it, the pipeline falls back to Claude Haiku.

Model download: ~500MB on first use, cached locally by HuggingFace.
Inference: ~50-100ms per headline on CPU.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level cache — model loaded once per process, reused across runs
_pipeline: Any = None
_available: bool | None = None   # None = not yet checked


def is_available() -> bool:
    """Return True if transformers + finbert model can be loaded."""
    global _available
    if _available is not None:
        return _available
    try:
        import transformers  # noqa: F401
        _available = True
    except ImportError:
        _available = False
        logger.info("[finbert] transformers not installed — using Haiku for sentiment")
    return _available


def classify_headlines(headlines: list[str]) -> list[dict[str, Any]]:
    """
    Classify a list of headlines using FinBERT.

    Parameters
    ----------
    headlines : list[str]
        Raw headline strings.

    Returns
    -------
    list of dicts: [{headline, sentiment, score}]
      sentiment: "positive" | "negative" | "neutral"
      score:     float from -1.0 (very negative) to +1.0 (very positive)

    Returns empty list if FinBERT is not available — caller must fall back.
    """
    if not is_available():
        return []

    try:
        pipeline = _load_pipeline()
        if pipeline is None:
            return []

        results = []
        for headline in headlines:
            try:
                pred   = pipeline(headline[:512])[0]   # Truncate to model max length
                label  = pred["label"].lower()          # "positive", "negative", "neutral"
                conf   = float(pred["score"])

                # Map to -1.0 → +1.0 scale
                if label == "positive":
                    score = conf
                elif label == "negative":
                    score = -conf
                else:
                    score = 0.0

                results.append({
                    "headline":  headline,
                    "sentiment": label,
                    "score":     round(score, 3),
                })
            except Exception as exc:
                logger.debug(f"[finbert] Failed on headline: {exc}")
                # Skip this headline rather than failing the whole batch
                continue

        return results

    except Exception as exc:
        logger.warning(f"[finbert] Classification error: {exc}")
        return []


def _load_pipeline() -> Any:
    """Lazy-load the FinBERT pipeline. Cached after first load."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    try:
        from transformers import pipeline
        logger.info("[finbert] Loading ProsusAI/finbert (first use — may take ~30s)...")
        _pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        logger.info("[finbert] Model loaded successfully")
        return _pipeline
    except Exception as exc:
        logger.warning(f"[finbert] Could not load model: {exc}")
        return None


def compute_aggregate_sentiment(
    classified: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Compute aggregate sentiment statistics from FinBERT classified headlines.

    Returns the same structure the News Agent LLM produces, so it can
    be used as a drop-in replacement for the LLM classification step.
    """
    if not classified:
        return {}

    total    = len(classified)
    pos      = sum(1 for h in classified if h["sentiment"] == "positive")
    neg      = sum(1 for h in classified if h["sentiment"] == "negative")
    neu      = total - pos - neg
    avg_score = sum(h["score"] for h in classified) / total

    if avg_score > 0.15:
        overall = "positive"
    elif avg_score < -0.15:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "overall_sentiment": overall,
        "sentiment_score":   round(avg_score, 3),
        "positive_pct":      round((pos / total) * 100, 1),
        "neutral_pct":       round((neu / total) * 100, 1),
        "negative_pct":      round((neg / total) * 100, 1),
        "top_headlines":     [
            {"headline": h["headline"], "sentiment": h["sentiment"], "score": h["score"]}
            for h in classified[:7]
        ],
    }
