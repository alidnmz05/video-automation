"""
Module 4: Asset Sourcer
------------------------
Downloads portrait (9:16) stock videos from the Pexels API
based on script keywords.
"""

import os
import logging
import requests

logger = logging.getLogger(__name__)

PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"


def download_assets(
    keywords: list,
    output_dir: str,
    api_key: str,
    clips_per_keyword: int = 2,
) -> list:
    """
    Search Pexels for portrait stock videos and download them.

    Args:
        keywords:          List of search terms (from script_generator).
        output_dir:        Directory to save downloaded .mp4 files.
        api_key:           Pexels API key.
        clips_per_keyword: How many clips to download per keyword (default 2).

    Returns:
        List of absolute paths to downloaded video files.
    """
    headers = {"Authorization": api_key}
    downloaded = []

    for keyword in keywords:
        logger.info(f"Searching Pexels: '{keyword}'")

        params = {
            "query": keyword,
            "orientation": "portrait",
            "size": "large",          # 1080p+
            "per_page": clips_per_keyword + 3,  # fetch extras, some may fail
        }

        try:
            resp = requests.get(
                PEXELS_SEARCH_URL, headers=headers, params=params, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning(f"Pexels search failed for '{keyword}': {e}")
            continue

        videos = data.get("videos", [])
        count = 0

        for video in videos:
            if count >= clips_per_keyword:
                break

            video_file = _pick_best_file(video.get("video_files", []))
            if not video_file:
                logger.debug(f"No suitable file found for video id={video.get('id')}")
                continue

            safe_kw = keyword.replace(" ", "_")[:25]
            filename = f"stock_{safe_kw}_{video['id']}.mp4"
            filepath = os.path.join(output_dir, filename)

            if _download(video_file["link"], filepath):
                downloaded.append(filepath)
                count += 1
                logger.info(f"  ✓ {filename} ({video_file.get('height', '?')}p)")

    if not downloaded:
        logger.warning(
            "No stock videos downloaded. Check your Pexels API key or try broader keywords."
        )

    return downloaded


# ── Helpers ──────────────────────────────────────────────────────────────────


def _pick_best_file(video_files: list) -> dict | None:
    """
    From a list of Pexels video file objects, pick the highest-resolution
    portrait file. Falls back to any file if no portrait file exists.
    """
    if not video_files:
        return None

    # Prefer portrait orientation (width < height)
    portrait = [f for f in video_files if f.get("width", 1) < f.get("height", 1)]
    candidates = portrait if portrait else video_files

    # Sort by height descending → highest resolution first
    candidates = sorted(candidates, key=lambda x: x.get("height", 0), reverse=True)
    return candidates[0]


def _download(url: str, filepath: str) -> bool:
    """Stream-download a file. Returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.RequestException as e:
        logger.error(f"Download failed for {url}: {e}")
        return False
