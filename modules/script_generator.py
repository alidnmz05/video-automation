"""
Module 1: Script Generator
--------------------------
Uses GPT-4o to produce a dark 40-second script and visual keywords
for stock footage search.
"""

import json
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a dark storytelling expert for short-form video content (TikTok, Reels, YouTube Shorts).
Your scripts are gripping, mysterious, and suspenseful.
You write in punchy, short sentences designed to hook viewers instantly."""

USER_PROMPT_TEMPLATE = """Create a gripping 40-second script about: "{topic}"

STRICT RULES:
- Maximum 8 words per sentence. Break long ideas into two sentences.
- Dark, mysterious, suspenseful tone throughout
- The very first sentence must be a killer hook
- No questions. No filler words. No "In this video..."
- Pure narrative or disturbing facts only
- Write ONLY in English

Also extract 4-6 visual keywords suitable for stock footage search on Pexels.
Keywords should be concrete and visual (e.g. "dark alley", "car crash", "hacker keyboard").

RESPOND IN THIS EXACT JSON FORMAT:
{{
  "script": "First punchy sentence. Second sentence. Continue the story. Keep it dark.",
  "keywords": ["keyword one", "keyword two", "keyword three", "keyword four"]
}}"""


def generate_script(topic: str, api_key: str) -> dict:
    """
    Generate a dark 40-second script and visual keywords using GPT-4o.

    Args:
        topic:   The subject of the video (e.g. "car crash conspiracy").
        api_key: OpenAI API key.

    Returns:
        dict with keys "script" (str) and "keywords" (list[str]).
    """
    client = OpenAI(api_key=api_key)

    logger.info(f"Generating script for topic: '{topic}'")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(topic=topic)},
        ],
        response_format={"type": "json_object"},
        temperature=0.92,
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    if "script" not in result or "keywords" not in result:
        raise ValueError(f"GPT-4o returned unexpected JSON structure: {raw}")

    logger.info(f"Script generated ({len(result['script'])} chars)")
    logger.info(f"Keywords: {result['keywords']}")

    return result
