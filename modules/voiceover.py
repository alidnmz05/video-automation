"""
Module 2: Voiceover Generator
------------------------------
Converts a script to speech using the ElevenLabs Text-to-Speech API
and saves the result as an MP3 file.
"""

import os
import logging
import requests

logger = logging.getLogger(__name__)

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


def generate_voiceover(
    text: str,
    output_dir: str,
    api_key: str,
    voice_id: str,
    model_id: str,
) -> str:
    """
    Generate an MP3 voiceover from text using ElevenLabs.

    Args:
        text:       The script to convert to speech.
        output_dir: Directory to save the MP3 file.
        api_key:    ElevenLabs API key.
        voice_id:   ElevenLabs voice ID.
        model_id:   ElevenLabs model ID (e.g. "eleven_monolingual_v1").

    Returns:
        Absolute path to the saved voiceover.mp3 file.
    """
    url = ELEVENLABS_TTS_URL.format(voice_id=voice_id)

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.80,
            "style": 0.50,
            "use_speaker_boost": True,
        },
    }

    logger.info(f"Requesting ElevenLabs TTS ({len(text)} chars, voice={voice_id})...")

    response = requests.post(url, json=payload, headers=headers, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs API error {response.status_code}: {response.text}"
        )

    audio_path = os.path.join(output_dir, "voiceover.mp3")
    with open(audio_path, "wb") as f:
        f.write(response.content)

    logger.info(f"Voiceover saved → {audio_path} ({len(response.content) / 1024:.1f} KB)")
    return audio_path
