#!/usr/bin/env python3
"""
Trend Pipeline Orchestrator
============================
Scans competitor channels → scores viral potential → rewrites content → produces video.

Usage:
    python trend_pipeline.py                        # Full automatic scan → produce
    python trend_pipeline.py --scan-only            # Analyse only, no video output
    python trend_pipeline.py --video-id VIDEO_ID    # Force-produce from a specific video
    python trend_pipeline.py --video-id VIDEO_ID --style conspiracy
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trend_pipeline")

CONFIG_FILE = Path(__file__).parent / "config.json"
CHANNELS_FILE = Path(__file__).parent / "channels.json"

BANNER = """
╔══════════════════════════════════════════════╗
║      🔥 Trend Detection Pipeline 🔥          ║
╚══════════════════════════════════════════════╝"""


# ── Config loaders ────────────────────────────────────────────────────────────


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        logger.error("config.json not found")
        sys.exit(1)
    with open(CONFIG_FILE) as f:
        return json.load(f)


def load_channels() -> dict:
    if not CHANNELS_FILE.exists():
        logger.error("channels.json not found")
        sys.exit(1)
    with open(CHANNELS_FILE) as f:
        return json.load(f)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print(BANNER)

    parser = argparse.ArgumentParser(description="Trend Detection Pipeline")
    parser.add_argument("--scan-only", action="store_true", help="Scan only, skip production")
    parser.add_argument("--video-id", type=str, help="Force-produce from a YouTube video ID")
    parser.add_argument("--style", type=str, default=None, help="Content style preset")
    args = parser.parse_args()

    config = load_config()
    channels_cfg = load_channels()

    yt_key = config["api_keys"].get("youtube", "").strip()
    threshold = channels_cfg["settings"].get("viral_threshold", 0.80)
    default_style = channels_cfg["settings"].get("default_style", "dark_mystery")
    style = args.style or default_style

    if not yt_key and not args.video_id:
        logger.error(
            "YouTube API key missing — add 'youtube' under api_keys in config.json"
        )
        sys.exit(1)

    # ── Force mode ────────────────────────────────────────────────────────────
    if args.video_id:
        logger.info(f"Force mode: producing from video {args.video_id}")
        produce_from_video(
            args.video_id,
            title=f"YouTube/{args.video_id}",
            style=style,
            config=config,
        )
        return

    # ── Step 1: Scan channels ─────────────────────────────────────────────────
    logger.info(f"[1/3] Scanning {len(channels_cfg['channels'])} channel(s)...")
    from modules.trend_detector import scan_channels, get_top_comments

    results = scan_channels(
        channels_cfg["channels"],
        yt_key,
        vph_threshold_ratio=channels_cfg["settings"].get("vph_alert_ratio", 1.5),
    )
    if not results:
        logger.info("No recent videos found.")
        return

    _print_table(results)

    if args.scan_only:
        logger.info("--scan-only set. Done.")
        return

    # ── Step 2: Score trending videos ─────────────────────────────────────────
    logger.info("[2/3] Scoring trending candidates...")
    from modules.viral_scorer import calculate_viral_score

    trending = [v for v in results if v["is_trending"]]
    if not trending:
        logger.info("No trending videos in this scan.")
        return

    best_video = None
    best_score = 0.0

    for video in trending:
        comments = get_top_comments(video["video_id"], yt_key, max_results=20)
        scores = calculate_viral_score(
            views=video["views"],
            likes=video["likes"],
            comments_count=video["comments"],
            vph_ratio=video["vph_ratio"],
            comment_texts=comments,
        )
        video["scores"] = scores
        decision = scores["decision"]
        logger.info(
            f"  [{decision}] {video['title'][:50]} — score {scores['viral_score']:.3f}"
        )
        if scores["viral_score"] > best_score:
            best_score = scores["viral_score"]
            best_video = video

    print()
    if best_score < threshold:
        logger.info(
            f"Best viral score {best_score:.3f} < threshold {threshold}. No video produced this run."
        )
        return

    # ── Step 3: Produce ───────────────────────────────────────────────────────
    logger.info(f"[3/3] 🚀 PRODUCING VIDEO — '{best_video['title']}'")
    produce_from_video(best_video["video_id"], best_video["title"], best_video["niche"], config)


# ── Production helper (also called from app.py) ───────────────────────────────


def produce_from_video(video_id: str, title: str, style: str, config: dict) -> str:
    """
    Full adapter → production run for a single video.

    Args:
        video_id: YouTube video ID.
        title:    Original video title (context for GPT).
        style:    Content style preset key.
        config:   Loaded config dict.

    Returns:
        Absolute path to the finished MP4.
    """
    from modules.content_adapter import fetch_transcript, rewrite_for_shorts
    from modules.voiceover import generate_voiceover
    from modules.subtitle_generator import generate_subtitles
    from modules.asset_sourcer import download_assets
    from modules.video_assembler import assemble_video

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = Path(config["temp_dir"]) / f"trend_{timestamp}"
    output_dir = Path(config["output_dir"])
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Fetch transcript (fall back to title-only if unavailable)
    try:
        transcript = fetch_transcript(video_id)
    except RuntimeError as exc:
        logger.warning(f"Transcript unavailable ({exc}). Using title as context.")
        transcript = f"A video about: {title}"

    # Rewrite
    result = rewrite_for_shorts(transcript, title, config["api_keys"]["openai"], style)
    script = result["script"]
    keywords = result["keywords"]

    logger.info(f"Script  : {script[:90]}{'...' if len(script) > 90 else ''}")
    logger.info(f"Keywords: {keywords}")
    (temp_dir / "script.txt").write_text(script, encoding="utf-8")

    # Production pipeline (identical to run.py steps 2–5)
    audio_path = generate_voiceover(
        script, str(temp_dir),
        config["api_keys"]["elevenlabs"],
        config["elevenlabs"]["voice_id"],
        config["elevenlabs"]["model_id"],
    )
    srt_path = generate_subtitles(audio_path, str(temp_dir), config["api_keys"]["openai"])
    clips = download_assets(keywords, str(temp_dir), config["api_keys"]["pexels"])
    output_path = str(output_dir / f"trend_{style}_{timestamp}.mp4")
    assemble_video(audio_path, clips, srt_path, output_path, config)

    print(f"\n✅  Video ready: {output_path}\n")
    return output_path


# ── CLI display ───────────────────────────────────────────────────────────────


def _print_table(results: list) -> None:
    print(f"\n{'─' * 72}")
    print(f"  {'CHANNEL':<18} {'TITLE':<35} {'VPH':>7} {'RATIO':>7}")
    print(f"{'─' * 72}")
    for r in results:
        flag = "🔥" if r["is_trending"] else "  "
        channel = r["channel_name"][:17]
        title = r["title"][:34]
        print(f"  {flag} {channel:<17} {title:<34} {r['vph']:>7.0f} {r['vph_ratio']:>6.2f}x")
    print(f"{'─' * 72}\n")


if __name__ == "__main__":
    main()
