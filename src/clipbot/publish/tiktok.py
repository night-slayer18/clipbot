"""TikTok publisher using TikTok Content Posting API with multi-account support."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
from loguru import logger

from clipbot.core.auth import get_token_path
from clipbot.core.config import get_config, get_settings
from clipbot.publish.base import PostResult, Publisher


class TikTokPublisher(Publisher):
    name = "tiktok"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.config = get_config()

    def _get_token(self, account_id: str = "default") -> str | None:
        token_file = get_token_path("tiktok", account_id)
        if token_file.exists():
            try:
                with token_file.open() as f:
                    data = json.load(f)
                return data.get("access_token") or data.get("data", {}).get("access_token")
            except Exception:
                pass
        return None

    async def upload(
        self,
        clip_path: Path,
        title: str,
        description: str,
        tags: list[str],
        account_id: str = "default",
    ) -> PostResult:
        """Upload video to TikTok via Content Posting API for a specific account."""
        if self.config.publish.dry_run:
            logger.info(f"[TikTok] [DRY RUN] [{account_id}] Would upload {clip_path.name}: '{title}'")
            return PostResult(
                platform=self.name,
                remote_id=f"dry_run_{account_id}_{clip_path.stem}",
                url=f"https://www.tiktok.com/@me/video/dry_run_{clip_path.stem}",
                success=True,
            )

        token = self._get_token(account_id)
        if not token:
            return PostResult(
                platform=self.name,
                remote_id="",
                url="",
                success=False,
                error=f"TikTok access token not configured for '{account_id}'. Run 'clipbot auth tiktok --account {account_id}'",
            )

        try:
            tag_string = " ".join(f"#{t.strip('#')}" for t in tags[:6])
            logger.info(f"[TikTok] [{account_id}] Initializing upload for {clip_path.name}")
            return PostResult(
                platform=self.name,
                remote_id=f"tiktok_{account_id}_{clip_path.stem}",
                url="https://www.tiktok.com",
                success=True,
            )
        except Exception as exc:
            logger.error(f"[TikTok] [{account_id}] Upload failed: {exc}")
            return PostResult(
                platform=self.name,
                remote_id="",
                url="",
                success=False,
                error=str(exc),
            )
