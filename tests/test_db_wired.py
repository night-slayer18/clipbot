"""Wired integration tests for real SQLite database operations and cascades."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import numpy as np
import pytest

from clipbot.core.models import Asset, AssetStatus, Clip, ClipStatus, Platform, Post, PostStatus, Source
from clipbot.core.repository import (
    get_approved_clips,
    get_asset_by_external,
    get_clip_by_id,
    get_posts_for_clip,
    get_signals,
    has_signals,
    insert_clip,
    insert_post,
    set_clip_decision,
    update_asset_status,
    update_clip,
    upsert_asset,
    upsert_signal,
)


@pytest.mark.asyncio
async def test_asset_lifecycle_and_conflict(test_db: aiosqlite.Connection, sample_asset: Asset):
    # 1. Insert asset
    await upsert_asset(test_db, sample_asset)
    
    # 2. Retrieve by external id
    fetched = await get_asset_by_external(test_db, sample_asset.source.value, sample_asset.external_id)
    assert fetched is not None
    assert fetched.id == sample_asset.id
    assert fetched.title == sample_asset.title
    assert fetched.status == AssetStatus.READY

    # 3. Conflict update (same source + external_id)
    sample_asset.status = AssetStatus.SCORED
    sample_asset.duration = 60.0
    await upsert_asset(test_db, sample_asset)

    updated = await get_asset_by_external(test_db, sample_asset.source.value, sample_asset.external_id)
    assert updated is not None
    assert updated.status == AssetStatus.SCORED
    assert updated.duration == 60.0


@pytest.mark.asyncio
async def test_signal_storage_and_unpacking(test_db: aiosqlite.Connection, sample_asset: Asset):
    await upsert_asset(test_db, sample_asset)

    # Pack float32 signals array
    original_arr = np.linspace(0.0, 1.0, 60, dtype=np.float32)
    await upsert_signal(test_db, sample_asset.id, "audio_rms", original_arr)

    assert await has_signals(test_db, sample_asset.id) is True
    
    signals = await get_signals(test_db, sample_asset.id)
    assert "audio_rms" in signals
    recovered_arr = signals["audio_rms"]
    assert np.allclose(original_arr, recovered_arr, atol=1e-5)


@pytest.mark.asyncio
async def test_clip_crud_and_status_transitions(test_db: aiosqlite.Connection, sample_asset: Asset, sample_clip: Clip):
    await upsert_asset(test_db, sample_asset)
    await insert_clip(test_db, sample_clip)

    # Fetch by ID
    clip = await get_clip_by_id(test_db, sample_clip.id)
    assert clip is not None
    assert clip.overall_score == sample_clip.overall_score
    assert clip.status == ClipStatus.APPROVED
    assert clip.tags == ["gta6", "gaming", "shorts"]

    # Review decision
    await set_clip_decision(test_db, sample_clip.id, "rejected", reason="poor_audio", in_delta=1.5, out_delta=-2.0)
    updated_clip = await get_clip_by_id(test_db, sample_clip.id)
    assert updated_clip is not None
    assert updated_clip.status == ClipStatus.REJECTED

    # Verify label was recorded
    async with test_db.execute("SELECT decision, reason, in_point_delta FROM labels WHERE clip_id=?", (sample_clip.id,)) as cur:
        row = await cur.fetchone()
        assert row is not None
        assert row[0] == "rejected"
        assert row[1] == "poor_audio"
        assert row[2] == 1.5


@pytest.mark.asyncio
async def test_post_lifecycle_and_isolation(test_db: aiosqlite.Connection, sample_asset: Asset, sample_clip: Clip):
    await upsert_asset(test_db, sample_asset)
    await insert_clip(test_db, sample_clip)

    post1 = Post(
        clip_id=sample_clip.id,
        platform=Platform.YOUTUBE_SHORTS,
        account_id="yt_main",
        remote_post_id="yt_video_001",
        status=PostStatus.PUBLISHED,
        published_at=datetime.now(timezone.utc),
    )
    post2 = Post(
        clip_id=sample_clip.id,
        platform=Platform.TIKTOK,
        account_id="tt_main",
        remote_post_id="tt_video_002",
        status=PostStatus.PUBLISHED,
        published_at=datetime.now(timezone.utc),
    )

    await insert_post(test_db, post1)
    await insert_post(test_db, post2)

    posts = await get_posts_for_clip(test_db, sample_clip.id)
    assert len(posts) == 2
    accounts = {p.account_id for p in posts}
    assert "yt_main" in accounts
    assert "tt_main" in accounts


@pytest.mark.asyncio
async def test_foreign_key_cascading_delete(test_db: aiosqlite.Connection, sample_asset: Asset, sample_clip: Clip):
    await upsert_asset(test_db, sample_asset)
    await insert_clip(test_db, sample_clip)
    await upsert_signal(test_db, sample_asset.id, "audio_rms", np.zeros(10, dtype=np.float32))

    post = Post(clip_id=sample_clip.id, platform=Platform.YOUTUBE_SHORTS, account_id="yt_main")
    await insert_post(test_db, post)

    # Deleting the parent asset should cascade-delete clips, signals, and posts
    await test_db.execute("DELETE FROM assets WHERE id=?", (sample_asset.id,))
    await test_db.commit()

    assert await get_asset_by_external(test_db, sample_asset.source.value, sample_asset.external_id) is None
    assert await get_clip_by_id(test_db, sample_clip.id) is None
    assert await has_signals(test_db, sample_asset.id) is False

    posts = await get_posts_for_clip(test_db, sample_clip.id)
    assert len(posts) == 0
