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

# ── Turkish style presets ─────────────────────────────────────────────────────
TR_STYLE_PRESETS = {
    "dark_mystery_tr": {
        "label": "🌑 Karanlık Gizem (TR)",
        "tone": "karanlık, gizemli ve rahatsız edici",
        "hook": "çoğu insanın bilmediği rahatsız edici bir gerçek veya şok edici bir açıklama",
        "format": "Türkçe Shorts anlatımı — saf karanlık atmosfer, sorular yok, kısa ve vurucu cümleler",
    },
    "educational_tr": {
        "label": "📚 Eğitici (TR)",
        "tone": "net, ilgi çekici ve gerçekten şaşırtıcı",
        "hook": "yaygın varsayımları sarsan akıl uçuran bir gerçek",
        "format": "Türkçe eğitici Shorts — anlaşılır, hızlı tempolu, samimi",
    },
    "conspiracy_tr": {
        "label": "🕵️ Komplo (TR)",
        "tone": "paranoyak, sorgulayıcı ve düşündürücü",
        "hook": "çoğu insanın hiç fark etmediği gizli bir gerçek veya şüpheli bir tesadüf",
        "format": "Türkçe komplo Shorts — şüphe uyandır, asla sonuca varma, izleyiciyi paranoyak bırak",
    },
    "tech_explained_tr": {
        "label": "💻 Teknoloji (TR)",
        "tone": "kesin, etkileyici ve hafifçe kaygı verici",
        "hook": "neredeyse imkânsız hissettiren bir teknik yetenek veya zafiyeti",
        "format": "Türkçe teknoloji açıklama Shorts — teknik terimler korunur, sade anlatılır",
    },
    "true_crime_tr": {
        "label": "🔪 Gerçek Suç (TR)",
        "tone": "ürpertici, metodolojik ve sürükleyici",
        "hook": "gerçek bir suç vakasının en rahatsız edici detayı",
        "format": "Türkçe gerçek suç Shorts — sinematik, gerçekçi, soğuk anlatım",
    },
    "motivational_tr": {
        "label": "🔥 Motivasyon (TR)",
        "tone": "yüksek enerjili, doğrudan ve acımasızca dürüst",
        "hook": "sert bir gerçek veya alışılmamış bir başarı ilkesi",
        "format": "Türkçe motivasyon Shorts — Hormozi tarzı, vurucu, gereksiz söz yok",
    },
}

# Birleşik preset erişimi (EN + TR)
ALL_PRESETS = {**STYLE_PRESETS, **TR_STYLE_PRESETS}

# ── EN rewrite prompt ─────────────────────────────────────────────────────────
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

# ── TR rewrite prompt ─────────────────────────────────────────────────────────
_TR_REWRITE_PROMPT = """\
Sen bir YouTube Shorts senaryo yazarısın. Türkçe içerik üretme konusunda uzmansın.

ORİJİNAL VİDEO BAŞLIĞI: {title}

TRANSCRIPT ÖZETİ (İngilizce kaynak):
{excerpt}

GÖREVİN:
1. Bu içeriğin TEMEL FİKRİNİ ve en çarpıcı gerçeklerini çıkar.
2. Bunu tamamen özgün bir 40 saniyelik Türkçe YouTube Shorts senaryosuna dönüştür.
3. Ton: {tone}
4. Kanca stratejisi: {hook}
5. Format: {format}

KATLANMAZ KURALLAR:
- Cümle başına maksimum 8 kelime. Kısa, çarpıcı, etkili.
- Transcript'ten doğrudan cümle kopyalama — fikri tamamen yeniden yorumla.
- Gerçek teknik terimleri, isimleri ve yerleri koru.
- Dolgu kelimeler yok. "Bu videoda..." gibi girişler yok. Soru ile bitirme.
- Senaryo başlı başına bir hikâye gibi hissettirmeli, özet gibi değil.
- SENARYO TAMAMEN TÜRKÇE OLMALI.

Pexels'te stok görüntü aramak için 4-6 somut görsel anahtar kelime çıkar.
Anahtar kelimeler İNGİLİZCE ve görsel olarak aranabilir olmalı (örn: "dark alley", "server room", "car crash").

TAM OLARAK BU JSON FORMATINDA CEVAP VER:
{{
  "script": "Tamamen yeniden yorumlanmış 40 saniyelik Türkçe senaryonuz burada.",
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
    language: str = "en",
) -> dict:
    """
    Rewrite a competitor transcript into an original short-form script via GPT-4o.

    Args:
        transcript:  Full transcript from fetch_transcript().
        video_title: Original video title (context for GPT).
        api_key:     OpenAI API key.
        style:       Style preset key (from STYLE_PRESETS or TR_STYLE_PRESETS).
        language:    "en" → English script, "tr" → Turkish script.

    Returns:
        dict  {"script": str, "keywords": list[str]}
        — same shape as script_generator.generate_script() for drop-in use.
        Note: keywords are always in English regardless of language (for Pexels).
    """
    from openai import OpenAI

    # Pick the right preset dict and prompt template based on language
    if language == "tr":
        preset_dict = TR_STYLE_PRESETS
        # Map EN style key → TR equivalent (e.g. "dark_mystery" → "dark_mystery_tr")
        tr_style = style if style.endswith("_tr") else f"{style}_tr"
        preset = preset_dict.get(tr_style, preset_dict["dark_mystery_tr"])
        prompt_template = _TR_REWRITE_PROMPT
    else:
        preset = STYLE_PRESETS.get(style, STYLE_PRESETS["dark_mystery"])
        prompt_template = _REWRITE_PROMPT

    # Limit transcript to ~2 000 chars to control token usage
    excerpt = transcript[:2000] + ("..." if len(transcript) > 2000 else "")

    prompt = prompt_template.format(
        title=video_title,
        excerpt=excerpt,
        tone=preset["tone"],
        hook=preset["hook"],
        format=preset["format"],
    )

    client = OpenAI(api_key=api_key)
    logger.info(f"Rewriting [{language.upper()}] in '{style}' style with GPT-4o ({len(excerpt)} chars input)...")

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
