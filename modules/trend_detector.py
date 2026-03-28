"""
Module: Trend Detector (Niche Spy)
------------------------------------
Monitors competitor YouTube channels via the YouTube Data API v3.
Calculates Views Per Hour (VPH) and flags videos whose VPH ratio
significantly exceeds the channel's historical average.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
CACHE_FILE = Path(__file__).parent.parent / "temp" / "trend_cache.json"


# ── Public API ────────────────────────────────────────────────────────────────


def scan_channels(
    channel_configs: list,
    api_key: str,
    vph_threshold_ratio: float = 1.5,
) -> list:
    """
    Scan all configured channels for trending videos.

    Flags a video as trending when its current VPH is ≥ vph_threshold_ratio
    times the channel's average VPH (derived from 4 older videos).

    Args:
        channel_configs:    List of {id, name, niche} dicts from channels.json.
        api_key:            YouTube Data API v3 key.
        vph_threshold_ratio: Minimum VPH multiplier to flag as trending.

    Returns:
        List of video dicts sorted by viral potential (trending first, then VPH ratio).
    """
    cache = _load_cache()
    results = []

    for channel in channel_configs:
        channel_id = channel["id"]
        channel_name = channel.get("name", channel_id)
        logger.info(f"Scanning channel: {channel_name}")

        try:
            video_ids = _get_recent_video_ids(channel_id, api_key, max_results=5)
            if not video_ids:
                logger.warning(f"  No videos found for {channel_name}")
                continue

            videos = _get_video_details(video_ids, api_key)
            if not videos:
                continue

            # Estimate channel average VPH from older videos (skip the newest)
            channel_avg_vph = _estimate_channel_avg_vph(videos[1:])

            # Analyse only the newest upload
            newest = videos[0]
            video_id = newest["id"]
            title = newest["snippet"]["title"]
            published_at = newest["snippet"]["publishedAt"]
            stats = newest.get("statistics", {})

            age_hours = _hours_since(published_at)
            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))
            comments = int(stats.get("commentCount", 0))

            # Lifetime VPH
            current_vph = views / max(age_hours, 0.001)

            # Delta VPH from cache (more accurate for repeat scans)
            cached = cache.get(video_id, {})
            if cached.get("views") and cached.get("timestamp"):
                prev_views = cached["views"]
                prev_ts = datetime.fromisoformat(cached["timestamp"])
                now = datetime.now(timezone.utc)
                delta_hours = (now - prev_ts).total_seconds() / 3600
                if delta_hours > 0.1:
                    current_vph = (views - prev_views) / delta_hours

            # Persist current state
            cache[video_id] = {
                "views": views,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            vph_ratio = current_vph / max(channel_avg_vph, 1.0)
            is_trending = (vph_ratio >= vph_threshold_ratio) and (age_hours <= 48)

            video_data = {
                "video_id": video_id,
                "title": title,
                "channel_id": channel_id,
                "channel_name": channel_name,
                "niche": channel.get("niche", "general"),
                "views": views,
                "likes": likes,
                "comments": comments,
                "age_hours": round(age_hours, 1),
                "vph": round(current_vph, 1),
                "channel_avg_vph": round(channel_avg_vph, 1),
                "vph_ratio": round(vph_ratio, 2),
                "is_trending": is_trending,
                "published_at": published_at,
                "url": f"https://youtube.com/watch?v={video_id}",
                "scores": None,  # filled later by viral_scorer
            }
            results.append(video_data)

            status = "🔥 TREND" if is_trending else "   —  "
            logger.info(
                f"  {status} | {title[:48]} | VPH {current_vph:.0f} ({vph_ratio:.2f}x avg)"
            )

        except requests.HTTPError as exc:
            logger.warning(f"HTTP error scanning {channel_name}: {exc}")
        except Exception as exc:
            logger.warning(f"Unexpected error scanning {channel_name}: {exc}")

    _save_cache(cache)
    results.sort(key=lambda x: (not x["is_trending"], -x["vph_ratio"]))
    return results


def get_top_comments(video_id: str, api_key: str, max_results: int = 20) -> list:
    """
    Fetch the most relevant comment texts for a video.

    Returns an empty list if comments are disabled or the API call fails
    (callers must handle this gracefully).
    """
    try:
        resp = requests.get(
            f"{YOUTUBE_API_BASE}/commentThreads",
            params={
                "part": "snippet",
                "videoId": video_id,
                "order": "relevance",
                "maxResults": max_results,
                "key": api_key,
                "textFormat": "plainText",
            },
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [
            item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            for item in items
        ]
    except Exception as exc:
        logger.debug(f"Comments unavailable for {video_id}: {exc}")
        return []


# ── Internal helpers ──────────────────────────────────────────────────────────


def _get_recent_video_ids(channel_id: str, api_key: str, max_results: int = 5) -> list:
    # Resolve uploads playlist
    resp = requests.get(
        f"{YOUTUBE_API_BASE}/channels",
        params={"part": "contentDetails", "id": channel_id, "key": api_key},
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        return []

    playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    resp = requests.get(
        f"{YOUTUBE_API_BASE}/playlistItems",
        params={
            "part": "contentDetails",
            "playlistId": playlist_id,
            "maxResults": max_results,
            "key": api_key,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return [item["contentDetails"]["videoId"] for item in resp.json().get("items", [])]


def _get_video_details(video_ids: list, api_key: str) -> list:
    resp = requests.get(
        f"{YOUTUBE_API_BASE}/videos",
        params={
            "part": "snippet,statistics",
            "id": ",".join(video_ids),
            "key": api_key,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("items", [])


def _estimate_channel_avg_vph(videos: list) -> float:
    """Average lifetime VPH across a list of video detail objects."""
    vphs = []
    for v in videos:
        views = int(v.get("statistics", {}).get("viewCount", 0))
        age_h = _hours_since(v["snippet"]["publishedAt"])
        if age_h > 0:
            vphs.append(views / age_h)
    return sum(vphs) / len(vphs) if vphs else 1_000.0


def _hours_since(iso_timestamp: str) -> float:
    published = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    return max((now - published).total_seconds() / 3600, 0.001)


def _load_cache() -> dict:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
