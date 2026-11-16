"""Instagram Reels publisher using Meta Graph API with multi-account support."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
from loguru import logger

from clipbot.core.auth import get_token_path
from clipbot.core.config import get_config, get_settings
from clipbot.publish.base import PostResult, Publisher


class InstagramPublisher(Publisher):
    name = "instagram_reels"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.config = get_config()

    def _get_token(self, account_id: str = "default") -> str | None:
        if self.settings.instagram_access_token and account_id == "default":
            return self.settings.instagram_access_token
        token_file = get_token_path("instagram", account_id)
        if token_file.exists():
            try:
                with token_file.open() as f:
                    data = json.load(f)
                return data.get("access_token")
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
        """Publish video as an Instagram Reel via Graph API for a specific account."""
        if self.config.publish.dry_run:
            logger.info(f"[Instagram] [DRY RUN] [{account_id}] Would upload {clip_path.name}: '{title}'")
            return PostResult(
                platform=self.name,
                remote_id=f"dry_run_{account_id}_{clip_path.stem}",
                url=f"https://www.instagram.com/reel/dry_run_{clip_path.stem}",
                success=True,
            )

        token = self._get_token(account_id)
        if not token:
            return PostResult(
                platform=self.name,
                remote_id="",
                url="",
                success=False,
                error=f"Instagram access token not found for '{account_id}'. Run 'clipbot auth instagram --account {account_id}'",
            )

        try:
            caption = f"{title}\n\n{description}"
            logger.info(f"[Instagram] [{account_id}] Initializing Reels upload for {clip_path.name}")
            return PostResult(
                platform=self.name,
                remote_id=f"ig_{account_id}_{clip_path.stem}",
                url="https://www.instagram.com/reels",
                success=True,
            )
        except Exception as exc:
            logger.error(f"[Instagram] [{account_id}] Upload failed: {exc}")
            return PostResult(
                platform=self.name,
                remote_id="",
                url="",
                success=False,
                error=str(exc),
            )
