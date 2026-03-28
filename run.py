#!/usr/bin/env python3
"""
AI-Driven Dark-Story Video Automator
======================================
Main entry point. Orchestrates all 5 modules to produce a
ready-to-upload 9:16 short-form video from a single topic string.

Usage:
    python run.py
    python run.py "car brake cut conspiracy"
"""

import json
import logging
import os
import sys
from datetime import datetime


# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("automator")

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

BANNER = """
╔══════════════════════════════════════════════╗
║      AI Dark-Story Video Automator           ║
╚══════════════════════════════════════════════╝
"""


# ── Config ────────────────────────────────────────────────────────────────────


def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        logger.error(f"config.json not found at: {CONFIG_FILE}")
        logger.info("Rename / copy config.json and fill in your API keys.")
        sys.exit(1)

    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

    # Validate API keys
    missing = [
        key
        for key in ("openai", "elevenlabs", "pexels")
        if not config["api_keys"].get(key)
    ]
    if missing:
        logger.error(
            f"Please fill in these API keys in config.json: {', '.join(missing)}"
        )
        sys.exit(1)

    return config


# ── Main pipeline ─────────────────────────────────────────────────────────────


def main() -> None:
    print(BANNER)

    config = load_config()

    # Topic: from CLI arg or interactive prompt
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:]).strip()
    else:
        topic = input("  Enter video topic (e.g. 'car brake cut conspiracy'): ").strip()

    if not topic:
        logger.error("Topic cannot be empty.")
        sys.exit(1)

    # Create session directory inside temp/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(config["temp_dir"], timestamp)
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)

    logger.info(f"Topic   : {topic}")
    logger.info(f"Session : {session_dir}")

    # ── Step 1: Script ────────────────────────────────────────────────────────
    _step(1, "Generating script (GPT-4o)...")
    from modules.script_generator import generate_script

    result = generate_script(topic, config["api_keys"]["openai"])
    script: str = result["script"]
    keywords: list = result["keywords"]

    logger.info(f"Script  : {script[:90]}{'...' if len(script) > 90 else ''}")
    logger.info(f"Keywords: {keywords}")

    # Save script to session dir for reference
    with open(os.path.join(session_dir, "script.txt"), "w", encoding="utf-8") as f:
        f.write(script)

    # ── Step 2: Voiceover ─────────────────────────────────────────────────────
    _step(2, "Generating voiceover (ElevenLabs)...")
    from modules.voiceover import generate_voiceover

    audio_path = generate_voiceover(
        script,
        session_dir,
        config["api_keys"]["elevenlabs"],
        config["elevenlabs"]["voice_id"],
        config["elevenlabs"]["model_id"],
    )

    # ── Step 3: Subtitles ─────────────────────────────────────────────────────
    _step(3, "Transcribing & generating subtitles (Whisper)...")
    from modules.subtitle_generator import generate_subtitles

    srt_path = generate_subtitles(audio_path, session_dir, config["api_keys"]["openai"])

    # ── Step 4: Stock videos ──────────────────────────────────────────────────
    _step(4, "Downloading stock videos (Pexels)...")
    from modules.asset_sourcer import download_assets

    stock_videos = download_assets(
        keywords, session_dir, config["api_keys"]["pexels"]
    )

    if not stock_videos:
        logger.warning(
            "No stock videos downloaded — video will render on a black background."
        )

    # ── Step 5: Assemble ──────────────────────────────────────────────────────
    _step(5, "Assembling final video (MoviePy)...")
    from modules.video_assembler import assemble_video

    output_filename = f"video_{timestamp}.mp4"
    output_path = os.path.join(config["output_dir"], output_filename)

    assemble_video(audio_path, stock_videos, srt_path, output_path, config)

    # ── Done ──────────────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║  ✅  DONE!                                   ║")
    print(f"║  📁  {output_path:<40}║")
    print("╚══════════════════════════════════════════════╝")
    print()


def _step(n: int, message: str) -> None:
    print()
    logger.info(f"[{n}/5] {message}")


if __name__ == "__main__":
    main()
