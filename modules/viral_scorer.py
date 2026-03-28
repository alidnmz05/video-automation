"""
Module: Viral Score Calculator
--------------------------------
Combines three signals into a single 0.0–1.0 viral score:
  - VPH Ratio   (50 %): How fast this video is growing vs channel average
  - Engagement  (30 %): (likes + comments) / views
  - Sentiment   (20 %): Emotional intensity of top comments (VADER)

Decision thresholds:
  ≥ 0.80  →  PRODUCE  (auto-trigger video production)
  ≥ 0.50  →  WATCH    (monitor for next scan)
  < 0.50  →  SKIP
"""

import logging
import math

logger = logging.getLogger(__name__)

W_VPH = 0.50
W_ENGAGEMENT = 0.30
W_SENTIMENT = 0.20


def calculate_viral_score(
    views: int,
    likes: int,
    comments_count: int,
    vph_ratio: float,
    comment_texts: list,
) -> dict:
    """
    Calculate multi-signal viral score for a video.

    Returns:
        dict with keys: viral_score, vph_score, engagement_score,
                        sentiment_score, decision
    """
    vph_s = _vph_score(vph_ratio)
    eng_s = _engagement_score(views, likes, comments_count)
    sent_s = _sentiment_score(comment_texts)

    viral_score = round(vph_s * W_VPH + eng_s * W_ENGAGEMENT + sent_s * W_SENTIMENT, 3)

    if viral_score >= 0.80:
        decision = "PRODUCE"
    elif viral_score >= 0.50:
        decision = "WATCH"
    else:
        decision = "SKIP"

    logger.info(
        f"Viral Score: {viral_score:.3f}  "
        f"[VPH={vph_s:.2f}  Eng={eng_s:.2f}  Sent={sent_s:.2f}]  → {decision}"
    )

    return {
        "viral_score": viral_score,
        "vph_score": round(vph_s, 3),
        "engagement_score": round(eng_s, 3),
        "sentiment_score": round(sent_s, 3),
        "decision": decision,
    }


# ── Signal calculators ────────────────────────────────────────────────────────


def _vph_score(vph_ratio: float) -> float:
    """
    Map VPH ratio → [0, 1] using a smooth sigmoid.
    ratio 1.0 → ~0.35,  ratio 2.0 → ~0.80,  ratio 3.0+ → ~1.0
    """
    # Logistic: score = 1 / (1 + e^(-k*(x - x0)))
    k = 2.0
    x0 = 2.0
    return min(1.0, 1.0 / (1.0 + math.exp(-k * (vph_ratio - x0))))


def _engagement_score(views: int, likes: int, comments: int) -> float:
    """
    engagement_rate = (likes + comments) / views
    Normalised: 5 % ER → 1.0  (typical high-performing video)
    """
    if views == 0:
        return 0.0
    er = (likes + comments) / views
    return min(1.0, er / 0.05)


def _sentiment_score(comment_texts: list) -> float:
    """
    Emotional intensity of comments using VADER.
    Both highly positive AND highly negative comments signal virality.
    abs(compound) averaged, then normalised.
    Returns 0.5 if VADER is unavailable (neutral assumption).
    """
    if not comment_texts:
        return 0.5

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        # Use absolute compound scores — strong emotion in any direction = viral
        intensities = [
            abs(analyzer.polarity_scores(t)["compound"]) for t in comment_texts
        ]
        return min(1.0, sum(intensities) / len(intensities))

    except ImportError:
        logger.warning("vaderSentiment not installed — sentiment score defaulting to 0.5")
        return 0.5
    except Exception as exc:
        logger.warning(f"Sentiment analysis failed: {exc}")
        return 0.5
