"""
Module: Content Adapter (LLM Re-Writer)
-----------------------------------------
1. Fetches a competitor video's transcript via youtube-transcript-api.
2. Rewrites the core idea into a 40-second original script using GPT-4o,
   applying the user's chosen content style preset.

Output format is identical to script_generator.generate_script() so the
full production pipeline (voiceover → subtitles → video) is plug-and-play.
"""

import json
import logging

logger = logging.getLogger(__name__)


# ── Style presets ─────────────────────────────────────────────────────────────
# Each preset controls tone, hook strategy, and format for the GPT-4o rewrite.
STYLE_PRESETS = {
    "dark_mystery": {
        "label":  "🌑 Dark Mystery",
        "tone":   "dark, mysterious, and deeply unsettling",
        "hook":   "a disturbing fact or shocking revelation that most people don't know",
        "format": "English Shorts narration — pure dark atmosphere, no questions, short punchy sentences",
    },
    "educational": {
        "label":  "📚 Educational",
        "tone":   "clear, engaging, and genuinely surprising",
        "hook":   "a mind-blowing fact that challenges common assumptions",
        "format": "English educational Shorts — accessible, fast-paced, friendly",
    },
    "conspiracy": {
        "label":  "🕵️ Conspiracy",
        "tone":   "paranoid, questioning, and thought-provoking",
        "hook":   "a hidden truth or suspicious coincidence most people never noticed",
        "format": "English conspiracy Shorts — raise doubt, never conclude, leave viewers paranoid",
    },
    "tech_explained": {
        "label":  "💻 Tech Explained",
        "tone":   "precise, impressive, and slightly ominous",
        "hook":   "a technical capability or exploit that feels almost impossible",
        "format": "English tech explainer Shorts — technical terms kept, explained simply",
    },
    "true_crime": {
        "label":  "🔪 True Crime",
        "tone":   "chilling, methodical, and gripping",
        "hook":   "the most disturbing detail of a real criminal case",
        "format": "English true crime Shorts — cinematic, factual, cold delivery",
    },
    "motivational": {
        "label":  "🔥 Motivational",
        "tone":   "high-energy, direct, and brutally honest",
        "hook":   "a harsh truth or counterintuitive success principle",
        "format": "English motivational Shorts — Hormozi-style, punchy, no fluff",
    },
}

_REWRITE_PROMPT = """\
You are an expert YouTube Shorts scriptwriter.

ORIGINAL VIDEO TITLE: {title}

TRANSCRIPT EXCERPT:
{excerpt}

YOUR MISSION:
1. Extract the CORE IDEA and most compelling facts from this content.
2. Reimagine it as a completely original 40-second YouTube Shorts script.
3. Tone: {tone}
4. Opening hook strategy: {hook}
5. Format: {format}

STRICT RULES:
- Maximum 8 words per sentence. Short, snappy, impactful.
- Do NOT copy phrases directly from the transcript — fully reimagine the idea.
- Keep any real technical terms, names, or places.
- No filler words. No "In this video...". No questions ending the script.
- The script must feel like a standalone story, not a summary.

Also extract 4–6 concrete visual keywords for stock footage search on Pexels.
Keywords must be highly visual and searchable (e.g. "dark alley", "server room", "car crash").

RESPOND IN THIS EXACT JSON FORMAT:
{{
  "script": "Your fully reimagined 40-second script here.",
  "keywords": ["keyword one", "keyword two", "keyword three", "keyword four"]
}}"""


# ── Public API ────────────────────────────────────────────────────────────────


def fetch_transcript(video_id: str, languages: list = None) -> str:
    """
    Fetch the caption transcript from a YouTube video.

    Falls back to auto-translated English if the preferred language is absent.

    Args:
        video_id:  11-character YouTube video ID.
        languages: Preferred language codes (default: ["en"]).

    Returns:
        Full transcript as a single plain text string.

    Raises:
        RuntimeError if no transcript is available.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        raise ImportError(
            "youtube-transcript-api is not installed. Run: pip install youtube-transcript-api"
        )

    if languages is None:
        languages = ["en"]

    logger.info(f"Fetching transcript for video: {video_id}")

    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        text = " ".join(seg["text"].strip() for seg in segments)
        logger.info(f"Transcript OK — {len(text)} chars")
        return text

    except Exception as primary_exc:
        logger.warning(f"Primary language failed ({primary_exc}), trying auto-translation...")
        try:
            listing = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = listing.find_generated_transcript(["en"])
            segments = transcript.fetch()
            return " ".join(seg["text"].strip() for seg in segments)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"No transcript available for {video_id}: {fallback_exc}"
            )


def rewrite_for_shorts(
    transcript: str,
    video_title: str,
    api_key: str,
    style: str = "dark_mystery",
) -> dict:
    """
    Rewrite a competitor transcript into an original short-form script via GPT-4o.

    Args:
        transcript:  Full transcript from fetch_transcript().
        video_title: Original video title (context for GPT).
        api_key:     OpenAI API key.
        style:       One of STYLE_PRESETS keys.

    Returns:
        dict  {"script": str, "keywords": list[str]}
        — same shape as script_generator.generate_script() for drop-in use.
    """
    from openai import OpenAI

    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["dark_mystery"])

    # Limit transcript to ~2 000 chars to control token usage
    excerpt = transcript[:2000] + ("..." if len(transcript) > 2000 else "")

    prompt = _REWRITE_PROMPT.format(
        title=video_title,
        excerpt=excerpt,
        tone=preset["tone"],
        hook=preset["hook"],
        format=preset["format"],
    )

    client = OpenAI(api_key=api_key)
    logger.info(f"Rewriting in '{style}' style with GPT-4o ({len(excerpt)} chars input)...")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.92,
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    if "script" not in result or "keywords" not in result:
        raise ValueError(f"GPT-4o returned unexpected JSON structure: {raw}")

    logger.info(f"Rewrite complete — {len(result['script'])} chars, keywords: {result['keywords']}")
    return result
