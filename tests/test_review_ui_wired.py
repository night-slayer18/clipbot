"""Wired integration tests for FastAPI review dashboard and byte-range streaming."""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import httpx
import pytest

from clipbot.core.config import get_config
from clipbot.core.db import init_db
from clipbot.core.models import Asset, Clip, ClipStatus
from clipbot.core.repository import insert_clip, upsert_asset
from clipbot.review.app import create_app
from tests.conftest import generate_real_mp4


@pytest.mark.asyncio
async def test_review_ui_endpoints_wired(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "review_test.db"
    
    # Configure custom test DB path
    config = get_config()
    orig_db_path = config.database.path
    config.database.path = str(db_path)
    # Allow open access in local tests
    monkeypatch.setenv("CLIPBOT_REVIEW_ALLOW_OPEN", "1")

    try:
        app = create_app()

        # Seed sample asset, video file, and clip
        video_file = tmp_path / "test_rendered.mp4"
        generate_real_mp4(video_file, duration_sec=3, fps=10)

        asset = Asset(
            id="rev-asset-1",
            title="Review Test Video",
            creator="Tester",
            url="https://youtube.com/watch?v=rev1",
        )
        clip = Clip(
            id="rev-clip-1",
            asset_id=asset.id,
            start_time=0.0,
            end_time=3.0,
            overall_score=0.95,
            rendered_clip_path=video_file,
            status=ClipStatus.RENDERED,
            title="Initial Title",
            description="Initial Desc",
            tags=["test", "review"],
        )

        await init_db(str(db_path))
        async with aiosqlite.connect(str(db_path)) as db:
            await upsert_asset(db, asset)
            await insert_clip(db, clip)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Health check
            res = await client.get("/health")
            assert res.status_code == 200
            assert res.json()["status"] == "ok"

            # 2. Stats
            res = await client.get("/api/stats")
            assert res.status_code == 200
            stats = res.json()
            assert stats["total_assets"] >= 1
            assert stats["pending_clips"] >= 1

            # 3. List clips
            res = await client.get("/api/clips?status=pending")
            assert res.status_code == 200
            clips = res.json()["clips"]
            assert len(clips) >= 1
            assert clips[0]["id"] == "rev-clip-1"

            # 4. Partial byte-range video streaming
            res = await client.get("/video/rev-clip-1", headers={"Range": "bytes=0-100"})
            assert res.status_code == 206
            assert "Content-Range" in res.headers
            assert len(res.content) == 101

            # 5. Approve clip
            res = await client.post(
                "/api/clips/rev-clip-1/approve",
                json={"title": "Approved Title", "description": "New Desc"},
            )
            assert res.status_code == 200
            assert res.json()["status"] == "approved"

            # 6. Reject clip
            res = await client.post(
                "/api/clips/rev-clip-1/reject",
                json={"reason": "too_boring"},
            )
            assert res.status_code == 200
            assert res.json()["status"] == "rejected"

    finally:
        config.database.path = orig_db_path
