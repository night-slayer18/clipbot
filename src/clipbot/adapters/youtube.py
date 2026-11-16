"""YouTube adapter — discovers GTA videos & highlights via API or yt-dlp search."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from clipbot.adapters.downloader import MediaDownloader
from clipbot.core.config import get_settings, load_sources
from clipbot.core.models import Asset, AssetStatus, Source


class YouTubeAdapter:
    name = "youtube"

    def __init__(self) -> None:
        self._settings = get_settings()

    async def discover(self, limit: int = 25) -> list[Asset]:
        """
        Discover GTA YouTube clips/highlights.
        Uses YouTube Data API v3 if key available, otherwise yt-dlp search.
        """
        sources = load_sources()
        cfg = sources.get("youtube", {})
        if not cfg.get("enabled", True):
            return []

        queries = cfg.get("search_queries", [
            "GTA RP highlights",
            "GTA 5 funny moments",
            "GTA RP best clips",
        ])

        api_key = self._settings.youtube_api_key
        if api_key:
            return await self._discover_via_api(queries, api_key, limit)
        else:
            return await self._discover_via_ytdlp(queries, limit)

    async def _discover_via_api(self, queries: list[str], api_key: str, limit: int) -> list[Asset]:
        logger.info("[YouTube] Discovering via YouTube Data API v3")
        assets: list[Asset] = []
        async with httpx.AsyncClient(timeout=20.0) as client:
            for query in queries:
                if len(assets) >= limit:
                    break
                try:
                    resp = await client.get(
                        "https://www.googleapis.com/youtube/v3/search",
                        params={
                            "part": "snippet",
                            "q": query,
                            "type": "video",
                            "videoDuration": "short",  # < 4 minutes
                            "maxResults": min(limit - len(assets), 25),
                            "order": "date",
                            "key": api_key,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for item in data.get("items", []):
                        vid_id = item["id"]["videoId"]
                        snippet = item["snippet"]
                        assets.append(
                            Asset(
                                source=Source.YOUTUBE,
                                external_id=vid_id,
                                creator=snippet.get("channelTitle", "YouTube Creator"),
                                title=snippet.get("title", ""),
                                url=f"https://www.youtube.com/watch?v={vid_id}",
                                status=AssetStatus.DISCOVERED,
                                metadata=snippet,
                            )
                        )
                except Exception as exc:
                    logger.warning(f"[YouTube] API search failed for '{query}': {exc}")
        return assets

    async def _discover_via_ytdlp(self, queries: list[str], limit: int) -> list[Asset]:
        """Discover using yt-dlp search (no API key required)."""
        logger.info("[YouTube] Discovering via yt-dlp search (no API key required)")
        assets: list[Asset] = []
        for query in queries:
            if len(assets) >= limit:
                break
            per_query_limit = max(1, min(10, limit - len(assets)))
            cmd = [
                "yt-dlp",
                f"ytsearch{per_query_limit}:{query}",
                "--dump-single-json",
                "--no-playlist",
                "--no-warnings",
            ]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await proc.communicate()
                if proc.returncode == 0:
                    data = json.loads(stdout.decode())
                    entries = data.get("entries", [])
                    for e in entries:
                        if not e:
                            continue
                        vid_id = e.get("id")
                        dur = e.get("duration", 0)
                        # Filter for reasonable clip length (under 5 minutes)
                        if dur and dur > 300:
                            continue
                        assets.append(
                            Asset(
                                source=Source.YOUTUBE,
                                external_id=vid_id,
                                creator=e.get("uploader", e.get("channel", "YouTube Creator")),
                                title=e.get("title", ""),
                                url=e.get("webpage_url") or f"https://www.youtube.com/watch?v={vid_id}",
                                duration=float(dur) if dur else None,
                                status=AssetStatus.DISCOVERED,
                                metadata=e,
                            )
                        )
            except Exception as exc:
                logger.warning(f"[YouTube] yt-dlp search failed for '{query}': {exc}")
        return assets

    async def download(self, asset: Asset, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_file = dest_dir / f"youtube_{asset.external_id}.mp4"
        return await MediaDownloader.download_media(asset.url, out_file)

    async def fetch_chat(self, asset: Asset, dest_dir: Path) -> Path | None:
        return None
