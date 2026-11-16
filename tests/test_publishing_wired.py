"""Wired integration tests for multi-account rate limiting and publishing dispatch."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest

from clipbot.core.config import get_config
from clipbot.core.models import Asset, Clip, ClipStatus, Platform, Post, PostStatus
from clipbot.core.repository import insert_clip, insert_post, upsert_asset
from clipbot.publish.manager import PublishManager
from clipbot.publish.rate_limiter import PublishRateLimiter
from tests.conftest import generate_real_mp4


@pytest.mark.asyncio
async def test_publish_rate_limiter_wired(tmp_path: Path):
    db_file = tmp_path / "rate_limit_test.db"
    from clipbot.core.db import init_db
    await init_db(str(db_file))

    limiter = PublishRateLimiter(str(db_file))

    # Initial state: should be allowed to post
    can_post, reason = await limiter.can_publish("youtube_shorts", account_id="yt_test")
    assert can_post is True

    # Record 4 published posts today for this account
    async with aiosqlite.connect(str(db_file)) as db:
        today_iso = datetime.now(timezone.utc).isoformat()
        for i in range(4):
            post = Post(
                clip_id=f"clip-{i}",
                platform=Platform.YOUTUBE_SHORTS,
                account_id="yt_test",
                status=PostStatus.PUBLISHED,
                published_at=datetime.now(timezone.utc),
            )
            await insert_post(db, post)

    # Now daily limit of 4 should be reached
    can_post, reason = await limiter.can_publish("youtube_shorts", account_id="yt_test")
    assert can_post is False
    assert "Daily limit reached" in reason

    # Different account should still be allowed
    can_post_other, _ = await limiter.can_publish("youtube_shorts", account_id="yt_secondary")
    assert can_post_other is True


@pytest.mark.asyncio
async def test_publish_manager_dry_run_dispatch_wired(tmp_path: Path):
    db_file = tmp_path / "dispatch_test.db"
    from clipbot.core.db import init_db
    await init_db(str(db_file))

    config = get_config()
    orig_dry_run = config.publish.dry_run
    config.publish.dry_run = True

    try:
        # Create real sample video file
        video_file = tmp_path / "clip_for_publish.mp4"
        generate_real_mp4(video_file, duration_sec=3)

        asset = Asset(id="pub-asset-1", title="Asset 1")
        clip = Clip(
            id="pub-clip-1",
            asset_id=asset.id,
            rendered_clip_path=video_file,
            status=ClipStatus.APPROVED,
            title="🔥 Insane Goal in 90th Minute!",
            description="Must watch finish",
            tags=["football", "goal", "skills"],
        )

        async with aiosqlite.connect(str(db_file)) as db:
            await upsert_asset(db, asset)
            await insert_clip(db, clip)

        mgr = PublishManager(str(db_file))
        # Dispatch approved clip
        dispatched_posts = await mgr.dispatch_approved_clips(limit=5)

        assert len(dispatched_posts) > 0
        assert all(p.status == PostStatus.PUBLISHED for p in dispatched_posts)

        # Verify posts recorded in SQLite
        async with aiosqlite.connect(str(db_file)) as db:
            async with db.execute("SELECT COUNT(*) FROM posts WHERE clip_id=?", (clip.id,)) as cur:
                count = (await cur.fetchone())[0]
                assert count == len(dispatched_posts)

    finally:
        config.publish.dry_run = orig_dry_run
