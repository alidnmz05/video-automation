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

TR_SYSTEM_PROMPT = """Sen kısa video içerikleri (TikTok, Reels, YouTube Shorts) için karanlık hikaye uzmanısın.
Senaryoların sürükleyici, gizemli ve gerilim doludur.
Her cümle izleyiciyi ekrana kilitlemelidir."""

TR_USER_PROMPT_TEMPLATE = """Şu konu hakkında 40 saniyelik sürükleyici bir Türkçe senaryo yaz: "{topic}"

KESIN KURALLAR:
- Cümle başına maksimum 8 kelime. Uzun fikirleri iki cümleye böl.
- Baştan sona karanlık, gizemli, gerilimli ton
- İlk cümle mutlaka izleyiciyi kilitleyen bir kanca olmalı
- Soru yok. Dolgu kelimesi yok. "Bu videoda..." yok.
- Yalnızca saf anlatım ya da rahatsız edici gerçekler
- Senaryoyu YALNIZCA TÜRKÇE yaz

Ayrıca Pexels'ta stok görüntü aramak için 4-6 görsel anahtar kelime çıkar.
Anahtar kelimeler somut ve görsel olmalı (ör. "dark alley", "car crash", "hacker keyboard").
Anahtar kelimeler MUTLAKA İNGİLİZCE olsun (Pexels için).

AYNEN BU JSON FORMATINDA CEVAP VER:
{{
  "script": "İlk güçlü Türkçe cümle. İkinci cümle. Hikayeye devam et. Karanlık tut.",
  "keywords": ["keyword one", "keyword two", "keyword three", "keyword four"]
}}"""


def generate_script(topic: str, api_key: str, language: str = "en") -> dict:
    """
    Generate a dark 40-second script and visual keywords using GPT-4o.

    Args:
        topic:    The subject of the video (e.g. "car crash conspiracy").
        api_key:  OpenAI API key.
        language: "en" (default) or "tr" for Turkish output.

    Returns:
        dict with keys "script" (str) and "keywords" (list[str]).
    """
    client = OpenAI(api_key=api_key)

    if language == "tr":
        system_prompt = TR_SYSTEM_PROMPT
        user_prompt   = TR_USER_PROMPT_TEMPLATE.format(topic=topic)
    else:
        system_prompt = SYSTEM_PROMPT
        user_prompt   = USER_PROMPT_TEMPLATE.format(topic=topic)

    logger.info(f"Generating [{language.upper()}] script for topic: '{topic}'")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
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
