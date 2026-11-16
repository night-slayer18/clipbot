"""YouTube Shorts publisher using YouTube Data API v3 with multi-account support."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from loguru import logger

from clipbot.core.auth import get_token_path
from clipbot.core.config import get_config, get_settings
from clipbot.publish.base import PostResult, Publisher


class YouTubePublisher(Publisher):
    name = "youtube_shorts"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.config = get_config()

    async def upload(
        self,
        clip_path: Path,
        title: str,
        description: str,
        tags: list[str],
        privacy_status: str = "public",
        account_id: str = "default",
    ) -> PostResult:
        """Upload a video file as a YouTube Short to a specific account."""
        if self.config.publish.dry_run:
            logger.info(f"[YouTube] [DRY RUN] [{account_id}] Would upload {clip_path.name}: '{title}'")
            return PostResult(
                platform=self.name,
                remote_id=f"dry_run_{account_id}_{clip_path.stem}",
                url=f"https://youtube.com/shorts/dry_run_{clip_path.stem}",
                success=True,
            )

        return await asyncio.get_event_loop().run_in_executor(
            None, self._upload_sync, clip_path, title, description, tags, privacy_status, account_id
        )

    def _upload_sync(
        self,
        clip_path: Path,
        title: str,
        description: str,
        tags: list[str],
        privacy_status: str,
        account_id: str,
    ) -> PostResult:
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload

            token_path = get_token_path("youtube", account_id)

            if not token_path.exists():
                return PostResult(
                    platform=self.name,
                    remote_id="",
                    url="",
                    success=False,
                    error=f"OAuth token not found at {token_path}. Run 'clipbot auth youtube --account {account_id}' first.",
                )

            creds = Credentials.from_authorized_user_file(str(token_path))
            youtube = build("youtube", "v3", credentials=creds)

            full_title = title if "#shorts" in title.lower() else f"{title} #shorts"
            full_desc = f"{description}\n\n#Shorts #Highlights"

            # Determine YouTube category ID dynamically
            combined_text = f"{title} {' '.join(tags)}".lower()
            if any(k in combined_text for k in ["nba", "basket", "cricket", "ipl", "football", "soccer", "ufc", "fight"]):
                category_id = "17"  # Sports
            elif any(k in combined_text for k in ["movie", "cinema", "film", "cartoon", "animation", "jerry"]):
                category_id = "1"   # Film & Animation
            elif any(k in combined_text for k in ["gta", "game", "gaming", "play", "twitch"]):
                category_id = "20"  # Gaming
            else:
                category_id = "24"  # Entertainment

            body = {
                "snippet": {
                    "title": full_title[:100],
                    "description": full_desc[:5000],
                    "tags": tags[:30],
                    "categoryId": category_id,
                },
                "status": {
                    "privacyStatus": privacy_status,
                    "selfDeclaredMadeForKids": False,
                },
            }

            media = MediaFileUpload(str(clip_path), chunksize=-1, resumable=True, mimetype="video/mp4")
            req = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

            logger.info(f"[YouTube] [{account_id}] Uploading {clip_path.name} to YouTube Shorts...")
            response = req.execute()
            video_id = response.get("id")

            logger.success(f"[YouTube] [{account_id}] Uploaded: https://youtube.com/shorts/{video_id}")
            return PostResult(
                platform=self.name,
                remote_id=video_id,
                url=f"https://youtube.com/shorts/{video_id}",
                success=True,
            )

        except Exception as exc:
            logger.error(f"[YouTube] [{account_id}] Upload failed: {exc}")
            return PostResult(
                platform=self.name,
                remote_id="",
                url="",
                success=False,
                error=str(exc),
            )
