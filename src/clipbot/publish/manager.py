"""Publish manager — coordinates approved clip dispatch across multiple accounts & platforms."""

from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
from loguru import logger

from clipbot.core.auth import load_accounts_config
from clipbot.core.config import get_config
from clipbot.core.models import ClipStatus, Platform, Post, PostStatus
from clipbot.core.repository import insert_post
from clipbot.publish.instagram import InstagramPublisher
from clipbot.publish.rate_limiter import PublishRateLimiter
from clipbot.publish.tiktok import TikTokPublisher
from clipbot.publish.youtube import YouTubePublisher


class PublishManager:
    """Dispatches approved clips to multiple accounts across YouTube Shorts, TikTok, and Instagram Reels."""

    def __init__(self, db_path: str = "data/state.db") -> None:
        self.db_path = db_path
        self.config = get_config()
        self.rate_limiter = PublishRateLimiter(db_path)

        self.publishers = {
            "youtube_shorts": YouTubePublisher(),
            "tiktok": TikTokPublisher(),
            "instagram_reels": InstagramPublisher(),
        }

    async def dispatch_approved_clips(
        self,
        limit: int = 3,
        platform_filter: str | None = None,
        account_filter: str | None = None,
    ) -> list[Post]:
        """Find approved clips and publish them across all configured accounts."""
        results: list[Post] = []
        accounts_cfg = load_accounts_config().get("accounts", {})
        anti_dup = load_accounts_config().get("anti_duplicate", {})

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT id, rendered_clip_path, title, description, tags_json, status
                FROM clips
                WHERE status IN ('approved', 'published') AND rendered_clip_path IS NOT NULL
                ORDER BY overall_score DESC
                LIMIT ?
                """,
                (limit * 2,),
            ) as cur:
                rows = await cur.fetchall()

        if not rows:
            logger.info("[Publish] No approved clips pending publication")
            return []

        async with aiosqlite.connect(self.db_path) as db:
            for clip_id, rendered_path, base_title, base_desc, tags_raw, cur_status in rows:
                if len(results) >= limit:
                    break

                clip_file = Path(rendered_path)
                if not clip_file.exists():
                    logger.warning(f"[Publish] Rendered file missing for clip {clip_id}: {rendered_path}")
                    continue

                tags = json.loads(tags_raw or "[]")
                target_platforms = [platform_filter] if platform_filter else list(self.publishers.keys())

                successful_uploads_count = 0

                for plat_name in target_platforms:
                    publisher = self.publishers.get(plat_name)
                    if not publisher:
                        continue

                    configured_accounts = accounts_cfg.get(plat_name, [{"id": "default", "enabled": True}])
                    if not configured_accounts:
                        configured_accounts = [{"id": "default", "enabled": True}]

                    for acc in configured_accounts:
                        acc_id = acc.get("id", "default")
                        if not acc.get("enabled", True):
                            continue
                        if account_filter and acc_id != account_filter:
                            continue

                        # Check if already published to this platform + account
                        async with db.execute(
                            "SELECT 1 FROM posts WHERE clip_id=? AND platform=? AND account_id=? AND status='published'",
                            (clip_id, plat_name, acc_id),
                        ) as check_cur:
                            if await check_cur.fetchone():
                                continue

                        # Rate limiter check for this specific account
                        can_post, reason = await self.rate_limiter.can_publish(plat_name, acc_id)
                        if not can_post:
                            logger.info(f"[Publish] Skipping {plat_name} [{acc_id}]: {reason}")
                            continue

                        # Anti-duplicate variation: scramble tags & title prefix per account
                        post_title = base_title or "Insane Highlight!"
                        post_tags = list(tags)

                        if anti_dup.get("scramble_metadata", True) and acc_id != "default":
                            random.shuffle(post_tags)
                            hook_variants = ["🔥", "😱", "💀", "⚡", "Watch this:"]
                            if not any(post_title.startswith(h) for h in hook_variants):
                                post_title = f"{random.choice(hook_variants)} {post_title}"

                        logger.info(f"[Publish] Dispatching {clip_id} -> {plat_name} [Account: {acc_id}]...")
                        post_res = await publisher.upload(
                            clip_path=clip_file,
                            title=post_title,
                            description=base_desc or "",
                            tags=post_tags,
                            account_id=acc_id,
                        )

                        post_status = PostStatus.PUBLISHED if post_res.success else PostStatus.FAILED
                        post = Post(
                            clip_id=clip_id,
                            platform=Platform(plat_name),
                            account_id=acc_id,
                            remote_post_id=post_res.remote_id,
                            status=post_status,
                            published_at=datetime.now(timezone.utc) if post_res.success else None,
                            error_message=post_res.error,
                        )
                        results.append(post)
                        await insert_post(db, post)

                        if post_res.success:
                            successful_uploads_count += 1

                if successful_uploads_count > 0 and cur_status != "published":
                    await db.execute("UPDATE clips SET status='published' WHERE id=?", (clip_id,))
                    await db.commit()

        logger.info(f"[Publish] Multi-account dispatch complete: {len(results)} posts processed.")
        return results
