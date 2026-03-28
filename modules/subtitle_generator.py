"""
Module 3: Subtitle Generator
-----------------------------
Transcribes the voiceover MP3 with OpenAI Whisper (word-level timestamps)
and outputs a Hormozi-style .srt file where words are grouped in pairs.
"""

import os
import logging
from datetime import timedelta

import srt
from openai import OpenAI

logger = logging.getLogger(__name__)


def generate_subtitles(audio_path: str, output_dir: str, api_key: str) -> str:
    """
    Transcribe audio with Whisper and produce a word-pair SRT file.

    Words are grouped in pairs (Hormozi/MrBeast style: 1-2 words at a time,
    large centred text, displayed exactly while spoken).

    Args:
        audio_path: Path to the voiceover MP3.
        output_dir: Directory to save subtitles.srt.
        api_key:    OpenAI API key (Whisper uses the same key).

    Returns:
        Absolute path to the saved .srt file.
    """
    client = OpenAI(api_key=api_key)

    logger.info(f"Transcribing audio with Whisper: {os.path.basename(audio_path)}")

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

    words = transcript.words
    if not words:
        raise ValueError("Whisper returned no word-level timestamps. Check the audio file.")

    logger.info(f"Whisper returned {len(words)} words with timestamps")

    # Group into pairs for Hormozi-style display
    subtitles = []
    i = 0
    idx = 1
    while i < len(words):
        group = words[i : i + 2]
        text = " ".join(w.word.strip() for w in group).upper()
        start = timedelta(seconds=float(group[0].start))
        end = timedelta(seconds=float(group[-1].end))

        # Ensure minimum display duration of 0.15 s
        if (end - start).total_seconds() < 0.15:
            end = start + timedelta(seconds=0.15)

        subtitles.append(
            srt.Subtitle(index=idx, start=start, end=end, content=text)
        )
        i += 2
        idx += 1

    srt_content = srt.compose(subtitles)
    srt_path = os.path.join(output_dir, "subtitles.srt")

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    logger.info(f"Generated {len(subtitles)} subtitle entries → {srt_path}")
    return srt_path
