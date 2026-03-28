#!/usr/bin/env python3
"""
AI-Driven Dark-Story Video Automator
======================================
Main entry point. Orchestrates all 5 modules to produce a
ready-to-upload 9:16 short-form video from a single topic string.

Usage:
    python run.py
    python run.py "car brake cut conspiracy"
    python run.py --lang tr "karanlık bir olay"
    python run.py --lang en --style conspiracy "NASA moon secrets"
"""

import argparse
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

    # CLI args
    parser = argparse.ArgumentParser(description="AI Video Automator")
    parser.add_argument("topic", nargs="*", help="Video topic")
    parser.add_argument("--lang", choices=["en", "tr"], default="en",
                        help="Output language: en (default) or tr")
    parser.add_argument("--style", default=None,
                        help="Content style preset (e.g. dark_mystery, conspiracy)")
    args = parser.parse_args()

    lang  = args.lang
    style = args.style

    if args.topic:
        topic = " ".join(args.topic).strip()
    else:
        lang_hint = "(TR — Türkçe senaryo üretilecek) " if lang == "tr" else ""
        topic = input(f"  Enter video topic {lang_hint}: ").strip()

    if not topic:
        logger.error("Topic cannot be empty.")
        sys.exit(1)

    # Create session directory inside temp/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(config["temp_dir"], timestamp)
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)

    logger.info(f"Topic   : {topic}")
    logger.info(f"Language: {lang.upper()}")
    logger.info(f"Session : {session_dir}")

    # ── Step 1: Script ────────────────────────────────────────────────────────
    _step(1, f"Generating script (GPT-4o) [{lang.upper()}]...")
    from modules.script_generator import generate_script

    result = generate_script(topic, config["api_keys"]["openai"], language=lang)
    script: str = result["script"]
    keywords: list = result["keywords"]

    logger.info(f"Script  : {script[:90]}{'...' if len(script) > 90 else ''}")
    logger.info(f"Keywords: {keywords}")

    # Save script to session dir for reference
    with open(os.path.join(session_dir, "script.txt"), "w", encoding="utf-8") as f:
        f.write(script)

    # ── Step 2: Voiceover ─────────────────────────────────────────────────────
    _step(2, f"Generating voiceover (ElevenLabs) [{lang.upper()}]...")
    from modules.voiceover import generate_voiceover

    el_cfg = config.get("elevenlabs_tr", config["elevenlabs"]) if lang == "tr" else config["elevenlabs"]
    audio_path = generate_voiceover(
        script,
        session_dir,
        config["api_keys"]["elevenlabs"],
        el_cfg["voice_id"],
        el_cfg["model_id"],
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

    output_filename = f"video_{lang}_{timestamp}.mp4"
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
