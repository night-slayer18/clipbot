"""Kick adapter — discovers GTA clips via Kick API, downloads via yt-dlp."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
from loguru import logger

from clipbot.adapters.downloader import MediaDownloader
from clipbot.core.config import get_settings, load_sources
from clipbot.core.models import Asset, AssetStatus, Source


KICK_API_BASE = "https://kick.com/api/v2"


class KickAdapter:
    name = "kick"

    def __init__(self) -> None:
        self._settings = get_settings()

    async def discover(self, limit: int = 30) -> list[Asset]:
        sources = load_sources()
        cfg = sources.get("kick", {})
        if not cfg.get("enabled", True):
            return []

        min_views = cfg.get("top_clips", {}).get("min_view_count", 100)
        categories = cfg.get("categories", ["grand-theft-auto-v"])
        assets: list[Asset] = []

        # Kick public clips API
        async with httpx.AsyncClient(
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
            follow_redirects=True,
            timeout=25,
        ) as client:
            for category in categories[:3]:
                try:
                    resp = await client.get(
                        f"{KICK_API_BASE}/clips",
                        params={"category": category, "sort": "view", "time": "day", "limit": limit},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        clips = data if isinstance(data, list) else data.get("clips", data.get("data", []))
                        for clip in clips[:limit]:
                            views = clip.get("views", clip.get("view_count", 0))
                            if views < min_views:
                                continue
                            clip_id = str(clip.get("id", clip.get("clip_id", "")))
                            url = clip.get("clip_url") or clip.get("url") or clip.get("video_url", "")
                            if not url:
                                continue
                            asset = Asset(
                                source=Source.KICK,
                                external_id=clip_id,
                                creator=clip.get("channel", {}).get("username", clip.get("streamer_username", "Kick Streamer")),
                                title=clip.get("title", ""),
                                url=f"https://kick.com/video/{clip_id}" if "kick.com" not in url else url,
                                duration=clip.get("duration"),
                                status=AssetStatus.DISCOVERED,
                                metadata=clip,
                            )
                            assets.append(asset)
                except Exception as exc:
                    logger.warning(f"[Kick] Discovery failed for category '{category}': {exc}")

        # Fallback to search if Kick API blocked by Cloudflare
        if not assets:
            logger.info("[Kick] Direct API returned 0 clips; attempting fallback search for Kick stream highlights")
            search_query = cfg.get("search_queries", ["Kick streamer highlights clips"])[0]
            cmd = [
                "yt-dlp",
                f"ytsearch{min(limit, 10)}:{search_query}",
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
                    import json
                    data = json.loads(stdout.decode())
                    for e in data.get("entries", []):
                        if not e:
                            continue
                        assets.append(
                            Asset(
                                source=Source.KICK,
                                external_id=f"yt_kick_{e.get('id')}",
                                creator=e.get("uploader", "Kick Streamer"),
                                title=e.get("title", ""),
                                url=e.get("webpage_url", ""),
                                duration=float(e.get("duration", 45)),
                                status=AssetStatus.DISCOVERED,
                                metadata=e,
                            )
                        )
            except Exception as exc:
                logger.warning(f"[Kick] Fallback search failed: {exc}")

        logger.info(f"[Kick] Discovered {len(assets)} clips")
        return assets

    async def download(self, asset: Asset, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        output = dest_dir / f"kick_{asset.external_id}.mp4"
        return await MediaDownloader.download_media(asset.url, output)

    async def fetch_chat(self, asset: Asset, dest_dir: Path) -> Path | None:
        return None
