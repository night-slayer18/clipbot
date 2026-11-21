"""Twitch adapter — discovers top GTA clips via Helix API or yt-dlp, downloads via yt-dlp."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
from loguru import logger

from clipbot.adapters.downloader import MediaDownloader
from clipbot.core.config import get_settings, load_sources
from clipbot.core.models import Asset, AssetStatus, Source


TWITCH_API_BASE = "https://api.twitch.tv/helix"


class TwitchAdapter:
    name = "twitch"

    def __init__(self) -> None:
        self._settings = get_settings()
        self._token: str | None = None
        self._token_file = Path("data/credentials/twitch_token.json")

    async def _get_token(self) -> str | None:
        if self._token:
            return self._token

        # Check saved user OAuth token
        if self._token_file.exists():
            try:
                with self._token_file.open() as f:
                    data = json.load(f)
                token = data.get("access_token")
                if token:
                    self._token = token
                    return self._token
            except Exception:
                pass

        # Otherwise use app client credentials
        if not self._settings.twitch_client_id or not self._settings.twitch_client_secret:
            return None
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    "https://id.twitch.tv/oauth2/token",
                    params={
                        "client_id": self._settings.twitch_client_id,
                        "client_secret": self._settings.twitch_client_secret,
                        "grant_type": "client_credentials",
                    },
                )
                resp.raise_for_status()
                self._token = resp.json()["access_token"]
            return self._token
        except Exception as exc:
            logger.warning(f"[Twitch] Failed to acquire OAuth token: {exc}")
            return None

    async def discover(self, limit: int = 50) -> list[Asset]:
        sources = load_sources()
        cfg = sources.get("twitch", {})
        if not cfg.get("enabled", True):
            return []

        token = await self._get_token()
        if token:
            return await self._discover_via_helix(token, cfg, limit)
        else:
            return await self._discover_via_fallback(cfg, limit)

    async def _discover_via_helix(self, token: str, cfg: dict, limit: int) -> list[Asset]:
        game_ids = cfg.get("game_ids", ["32982"])
        min_views = cfg.get("top_clips", {}).get("min_view_count", 300)
        period = cfg.get("top_clips", {}).get("period", "day")
        assets: list[Asset] = []

        async with httpx.AsyncClient(timeout=15.0) as client:
            for game_id in game_ids:
                try:
                    resp = await client.get(
                        f"{TWITCH_API_BASE}/clips",
                        params={
                            "game_id": game_id,
                            "first": min(limit, 100),
                            "started_at": self._period_to_iso(period),
                        },
                        headers={
                            "Client-Id": self._settings.twitch_client_id,
                            "Authorization": f"Bearer {token}",
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for clip in data.get("data", []):
                        if clip.get("view_count", 0) < min_views:
                            continue
                        asset = Asset(
                            source=Source.TWITCH,
                            external_id=clip["id"],
                            creator=clip.get("broadcaster_name", "Twitch Streamer"),
                            title=clip.get("title", ""),
                            url=clip["url"],
                            duration=clip.get("duration"),
                            status=AssetStatus.DISCOVERED,
                            metadata=clip,
                        )
                        assets.append(asset)
                except Exception as exc:
                    logger.warning(f"[Twitch] Helix clips request failed: {exc}")

        logger.info(f"[Twitch] Discovered {len(assets)} clips via Helix API")
        return assets

    async def _discover_via_fallback(self, cfg: dict, limit: int) -> list[Asset]:
        """Fallback when no Twitch API keys provided: yt-dlp search for trending clips."""
        queries = cfg.get("search_queries", ["Twitch viral highlights clips"])
        query = queries[0] if queries else "Twitch highlights clips"
        logger.info(f"[Twitch] No API credentials found; using fallback search for: '{query}'")
        cmd = [
            "yt-dlp",
            f"ytsearch{min(limit, 10)}:{query}",
            "--dump-single-json",
            "--no-playlist",
            "--no-warnings",
        ]
        assets: list[Asset] = []
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                data = json.loads(stdout.decode())
                for e in data.get("entries", []):
                    if not e:
                        continue
                    assets.append(
                        Asset(
                            source=Source.TWITCH,
                            external_id=f"yt_twitch_{e.get('id')}",
                            creator=e.get("uploader", "Twitch Streamer"),
                            title=e.get("title", ""),
                            url=e.get("webpage_url", ""),
                            duration=float(e.get("duration", 45)),
                            status=AssetStatus.DISCOVERED,
                            metadata=e,
                        )
                    )
        except Exception as exc:
            logger.warning(f"[Twitch] Fallback discovery failed: {exc}")
        return assets

    async def download(self, asset: Asset, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        output = dest_dir / f"twitch_{asset.external_id}.mp4"
        return await MediaDownloader.download_media(asset.url, output)

    async def fetch_chat(self, asset: Asset, dest_dir: Path) -> Path | None:
        """Fetch Twitch VOD chat comments replay using yt-dlp comment extractor."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        chat_output = dest_dir / f"twitch_{asset.external_id}.chat.json"
        if chat_output.exists() and chat_output.stat().st_size > 50:
            return chat_output

        cmd = [
            "yt-dlp",
            "--write-comments",
            "--skip-download",
            "-o", str(dest_dir / f"twitch_{asset.external_id}"),
            asset.url,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
            )
            await asyncio.wait_for(proc.wait(), timeout=15.0)
            # yt-dlp outputs comments in .info.json or .comments.json
            comments_file = dest_dir / f"twitch_{asset.external_id}.comments.json"
            info_file = dest_dir / f"twitch_{asset.external_id}.info.json"
            if comments_file.exists() and comments_file.stat().st_size > 50:
                return comments_file
            if info_file.exists() and info_file.stat().st_size > 50:
                return info_file
        except Exception as exc:
            logger.debug(f"[Twitch] Chat comment extraction notice for {asset.external_id}: {exc}")

        return None

    @staticmethod
    def _period_to_iso(period: str) -> str:
        from datetime import datetime, timedelta, timezone
        deltas = {"day": 1, "week": 7, "month": 30}
        days = deltas.get(period, 1)
        dt = datetime.now(timezone.utc) - timedelta(days=days)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
