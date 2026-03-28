"""
Module 5: Video Assembler
--------------------------
The core rendering engine. Uses MoviePy + Pillow to:
  1. Crop & resize stock clips to 9:16 (1080×1920)
  2. Sequence clips to match audio length
  3. Add a dark semi-transparent overlay
  4. Render Hormozi-style word-pair subtitles from the SRT file
  5. Mux with the ElevenLabs voiceover and export as H.264/AAC MP4
"""

import math
import os
import logging
import platform

import numpy as np
import srt as srtlib
from PIL import Image, ImageDraw, ImageFont, ImageColor
from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────


def assemble_video(
    audio_path: str,
    video_paths: list,
    srt_path: str,
    output_path: str,
    config: dict,
) -> str:
    """
    Assemble the final vertical short-form video.

    Args:
        audio_path:  Path to the ElevenLabs MP3 voiceover.
        video_paths: List of downloaded stock video paths.
        srt_path:    Path to the Whisper-generated SRT file.
        output_path: Where to save the finished MP4.
        config:      Full config dict (video, typography sections used).

    Returns:
        output_path on success.
    """
    W: int = config["video"]["width"]    # 1080
    H: int = config["video"]["height"]   # 1920
    FPS: int = config["video"]["fps"]    # 30
    overlay_opacity: float = config["video"]["overlay_opacity"]

    # ── 1. Audio ──────────────────────────────────────────────────────────────
    logger.info("Loading audio...")
    audio = AudioFileClip(audio_path)
    total_duration: float = audio.duration
    logger.info(f"Total duration: {total_duration:.2f}s")

    # ── 2. Base video ─────────────────────────────────────────────────────────
    if video_paths:
        logger.info(f"Building base video from {len(video_paths)} clip(s)...")
        base_video = _build_base_video(video_paths, total_duration, W, H)
    else:
        logger.warning("No stock videos — using solid black background.")
        base_video = ColorClip(size=(W, H), color=[0, 0, 0], duration=total_duration)

    # ── 3. Dark overlay ───────────────────────────────────────────────────────
    overlay = (
        ColorClip(size=(W, H), color=[0, 0, 0], duration=total_duration)
        .set_opacity(overlay_opacity)
    )

    # ── 4. Subtitle clips ─────────────────────────────────────────────────────
    logger.info("Rendering subtitle clips...")
    font_path = _find_font(config["typography"]["font"])
    subtitle_clips = _build_subtitle_clips(srt_path, (W, H), config["typography"], font_path)
    logger.info(f"Created {len(subtitle_clips)} subtitle clip(s)")

    # ── 5. Composite & export ─────────────────────────────────────────────────
    logger.info("Compositing layers...")
    final = CompositeVideoClip(
        [base_video, overlay] + subtitle_clips,
        size=(W, H),
    )
    final = final.set_audio(audio).set_duration(total_duration)

    logger.info(f"Exporting → {output_path}")
    final.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="ultrafast",
        logger="bar",
    )

    audio.close()
    final.close()

    return output_path


# ── Base video helpers ────────────────────────────────────────────────────────


def _build_base_video(video_paths: list, total_duration: float, W: int, H: int):
    """
    Load each stock clip, crop to 9:16, resize to (W, H), loop/trim to
    an equal share of total_duration, then concatenate.
    """
    n = len(video_paths)
    clip_share = total_duration / n

    processed = []
    for i, path in enumerate(video_paths):
        logger.info(f"  Processing clip {i + 1}/{n}: {os.path.basename(path)}")
        try:
            clip = VideoFileClip(path, audio=False)
        except Exception as exc:
            logger.warning(f"  Could not load {path}: {exc} — skipping")
            continue

        clip = _crop_to_ratio(clip, W, H).resize((W, H))

        # Loop if the clip is shorter than required share
        if clip.duration < clip_share:
            loops = math.ceil(clip_share / clip.duration)
            clip = concatenate_videoclips([clip] * loops)

        clip = clip.subclip(0, min(clip_share, clip.duration))
        processed.append(clip)

    if not processed:
        logger.warning("All stock clips failed to load — using black background.")
        return ColorClip(size=(W, H), color=[0, 0, 0], duration=total_duration)

    # If some clips were skipped, redistribute remaining duration evenly
    if len(processed) < n:
        new_share = total_duration / len(processed)
        redistributed = []
        for clip in processed:
            if clip.duration < new_share:
                loops = math.ceil(new_share / clip.duration)
                clip = concatenate_videoclips([clip] * loops)
            redistributed.append(clip.subclip(0, new_share))
        processed = redistributed

    return concatenate_videoclips(processed, method="compose")


def _crop_to_ratio(clip, W: int, H: int):
    """Centre-crop a VideoFileClip to the W:H aspect ratio."""
    target_ratio = W / H
    clip_ratio = clip.w / clip.h

    if abs(clip_ratio - target_ratio) < 0.02:
        return clip  # Close enough, skip crop

    if clip_ratio > target_ratio:
        # Clip is wider → crop left and right
        new_w = int(clip.h * target_ratio)
        x1 = int((clip.w - new_w) / 2)
        return clip.crop(x1=x1, x2=x1 + new_w)
    else:
        # Clip is taller → crop top and bottom
        new_h = int(clip.w / target_ratio)
        y1 = int((clip.h - new_h) / 2)
        return clip.crop(y1=y1, y2=y1 + new_h)


# ── Subtitle helpers ──────────────────────────────────────────────────────────


def _build_subtitle_clips(
    srt_path: str,
    video_size: tuple,
    typography: dict,
    font_path: str | None,
) -> list:
    """Parse the SRT file and produce a transparent ImageClip per entry."""
    with open(srt_path, "r", encoding="utf-8") as f:
        subs = list(srtlib.parse(f.read()))

    W, H = video_size
    clips = []

    for sub in subs:
        start = sub.start.total_seconds()
        duration = sub.end.total_seconds() - start
        text = sub.content.strip()

        if not text or duration <= 0:
            continue

        frame = _render_text_frame(text, W, H, font_path, typography)
        clip = ImageClip(frame, duration=duration).set_start(start)
        clips.append(clip)

    return clips


def _render_text_frame(
    text: str,
    W: int,
    H: int,
    font_path: str | None,
    typography: dict,
) -> np.ndarray:
    """
    Render a single subtitle line onto a transparent RGBA canvas using Pillow.
    Returns a numpy array suitable for MoviePy's ImageClip.
    """
    font_size: int = typography["font_size"]
    color: str = typography["font_color"]
    stroke_color: str = typography["stroke_color"]
    stroke_width: int = typography["stroke_width"]

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load font
    font = _load_font(font_path, font_size)

    # Parse named colours → RGBA tuples
    main_rgba = _to_rgba(color)
    stroke_rgba = _to_rgba(stroke_color)

    # Centre the text horizontally, place at 75 % height
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (W - text_w) / 2 - bbox[0]
    y = H * 0.75 - text_h / 2 - bbox[1]

    draw.text(
        (x, y),
        text,
        font=font,
        fill=main_rgba,
        stroke_width=stroke_width,
        stroke_fill=stroke_rgba,
    )

    return np.array(img)


def _load_font(font_path: str | None, size: int):
    """Load a TrueType font, falling back to PIL default if not found."""
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception as exc:
            logger.warning(f"Could not load font at {font_path}: {exc}")
    logger.warning(
        "Using PIL default font — install Impact for best results. "
        "See README for font setup."
    )
    return ImageFont.load_default()


def _to_rgba(color) -> tuple:
    """Convert a CSS colour name or RGB tuple to an RGBA tuple."""
    if isinstance(color, str):
        rgb = ImageColor.getrgb(color)
    else:
        rgb = tuple(color)
    return rgb + (255,) if len(rgb) == 3 else tuple(rgb)


# ── Font discovery ────────────────────────────────────────────────────────────


def _find_font(font_name: str) -> str | None:
    """
    Locate a system font by name. Returns the file path or None.
    Checks known paths first, then scans common font directories.
    """
    system = platform.system()

    # Known direct paths per font per OS
    known = {
        "Impact": {
            "Windows": ["C:\\Windows\\Fonts\\impact.ttf"],
            "Darwin": ["/Library/Fonts/Impact.ttf"],
            "Linux": [
                "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
                "/usr/share/fonts/truetype/impact.ttf",
            ],
        },
        "Montserrat": {
            "Windows": [
                "C:\\Windows\\Fonts\\Montserrat-Bold.ttf",
                "C:\\Windows\\Fonts\\Montserrat-ExtraBold.ttf",
            ],
            "Darwin": ["/Library/Fonts/Montserrat-Bold.ttf"],
            "Linux": [
                "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
            ],
        },
    }

    for path in known.get(font_name, {}).get(system, []):
        if os.path.exists(path):
            logger.info(f"Font found: {path}")
            return path

    # Fallback: scan system font directories
    font_dirs = {
        "Windows": ["C:\\Windows\\Fonts"],
        "Darwin": ["/Library/Fonts", "/System/Library/Fonts"],
        "Linux": ["/usr/share/fonts", "/usr/local/share/fonts"],
    }.get(system, [])

    search_name = font_name.lower().replace(" ", "")
    for font_dir in font_dirs:
        if not os.path.isdir(font_dir):
            continue
        for filename in os.listdir(font_dir):
            if search_name in filename.lower() and filename.lower().endswith(".ttf"):
                full = os.path.join(font_dir, filename)
                logger.info(f"Font found via scan: {full}")
                return full

    logger.warning(f"Font '{font_name}' not found. Subtitles will use PIL default font.")
    return None
