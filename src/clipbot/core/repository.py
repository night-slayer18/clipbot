"""Database repository — read/write helpers for all core models."""

from __future__ import annotations

import json
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
import numpy as np
from loguru import logger

from clipbot.core.models import (
    Asset, AssetStatus, Clip, ClipStatus, Post, PostStatus,
    Platform, ReviewLabel, Source,
)


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _pack_array(arr: np.ndarray) -> bytes:
    """Compress float32 array → zlib bytes for storage."""
    return zlib.compress(arr.astype(np.float32).tobytes(), level=1)


def _unpack_array(blob: bytes) -> np.ndarray:
    return np.frombuffer(zlib.decompress(blob), dtype=np.float32)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Asset CRUD ────────────────────────────────────────────────────────────────

async def upsert_asset(db: aiosqlite.Connection, asset: Asset) -> str:
    """Insert or update an asset row (keyed on source + external_id).

    Ensures asset.id is synchronized with the canonical database primary key.
    """
    await db.execute(
        """
        INSERT INTO assets
            (id, source, external_id, creator, title, game, url, duration,
             local_path, chat_path, status, metadata_json, created_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(source, external_id) DO UPDATE SET
            status=excluded.status,
            local_path=excluded.local_path,
            chat_path=excluded.chat_path,
            duration=excluded.duration,
            updated_at=excluded.updated_at
        """,
        (
            asset.id, asset.source.value, asset.external_id, asset.creator,
            asset.title, asset.game, asset.url, asset.duration,
            str(asset.local_path) if asset.local_path else None,
            str(asset.chat_path) if asset.chat_path else None,
            asset.status.value,
            json.dumps(asset.metadata),
            asset.created_at.isoformat(),
            _now(),
        ),
    )
    async with db.execute(
        "SELECT id FROM assets WHERE source=? AND external_id=?",
        (asset.source.value, asset.external_id),
    ) as cur:
        row = await cur.fetchone()
        if row:
            asset.id = row[0]
    await db.commit()
    return asset.id


async def get_asset_by_external(
    db: aiosqlite.Connection, source: str, external_id: str
) -> Optional[Asset]:
    async with db.execute(
        "SELECT * FROM assets WHERE source=? AND external_id=?",
        (source, external_id),
    ) as cur:
        row = await cur.fetchone()
    if not row:
        return None
    return _row_to_asset(row)


async def update_asset_status(
    db: aiosqlite.Connection, asset_id: str, status: AssetStatus
) -> None:
    await db.execute(
        "UPDATE assets SET status=?, updated_at=? WHERE id=?",
        (status.value, _now(), asset_id),
    )
    await db.commit()


def _row_to_asset(row) -> Asset:
    cols = [
        "id", "source", "external_id", "creator", "title", "game", "url",
        "duration", "local_path", "chat_path", "status", "metadata_json",
        "created_at", "updated_at",
    ]
    d = dict(zip(cols, row))
    return Asset(
        id=d["id"],
        source=Source(d["source"]),
        external_id=d["external_id"],
        creator=d["creator"],
        title=d["title"] or "",
        game=d["game"] or "General",
        url=d["url"],
        duration=d["duration"],
        local_path=Path(d["local_path"]) if d["local_path"] else None,
        chat_path=Path(d["chat_path"]) if d["chat_path"] else None,
        status=AssetStatus(d["status"]),
        metadata=json.loads(d["metadata_json"] or "{}"),
    )


# ── Signal CRUD ───────────────────────────────────────────────────────────────

async def upsert_signal(
    db: aiosqlite.Connection, asset_id: str, detector: str, arr: np.ndarray
) -> None:
    blob = _pack_array(arr)
    await db.execute(
        """
        INSERT INTO signals (asset_id, detector, series_blob)
        VALUES (?,?,?)
        ON CONFLICT(asset_id, detector) DO UPDATE SET series_blob=excluded.series_blob
        """,
        (asset_id, detector, blob),
    )
    await db.commit()


async def get_signals(
    db: aiosqlite.Connection, asset_id: str
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    async with db.execute(
        "SELECT detector, series_blob FROM signals WHERE asset_id=?", (asset_id,)
    ) as cur:
        async for row in cur:
            result[row[0]] = _unpack_array(row[1])
    return result


async def has_signals(db: aiosqlite.Connection, asset_id: str) -> bool:
    async with db.execute(
        "SELECT COUNT(*) FROM signals WHERE asset_id=?", (asset_id,)
    ) as cur:
        row = await cur.fetchone()
    return bool(row and row[0] > 0)


# ── Clip CRUD ─────────────────────────────────────────────────────────────────

async def insert_clip(db: aiosqlite.Connection, clip: Clip) -> None:
    await db.execute(
        """
        INSERT OR IGNORE INTO clips
            (id, asset_id, start_time, end_time, overall_score, scores_breakdown,
             raw_clip_path, rendered_clip_path, status, title, description, tags_json, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            clip.id, clip.asset_id, clip.start_time, clip.end_time,
            clip.overall_score, json.dumps(clip.scores_breakdown),
            str(clip.raw_clip_path) if clip.raw_clip_path else None,
            str(clip.rendered_clip_path) if clip.rendered_clip_path else None,
            clip.status.value, clip.title, clip.description,
            json.dumps(clip.tags),
            clip.created_at.isoformat(),
        ),
    )
    await db.commit()


async def update_clip(db: aiosqlite.Connection, clip: Clip) -> None:
    await db.execute(
        """
        UPDATE clips SET
            status=?, raw_clip_path=?, rendered_clip_path=?,
            title=?, description=?, tags_json=?
        WHERE id=?
        """,
        (
            clip.status.value,
            str(clip.raw_clip_path) if clip.raw_clip_path else None,
            str(clip.rendered_clip_path) if clip.rendered_clip_path else None,
            clip.title, clip.description, json.dumps(clip.tags),
            clip.id,
        ),
    )
    await db.commit()


async def get_clips_for_review(db: aiosqlite.Connection) -> list[dict]:
    """Return rendered clips awaiting review, richest first."""
    async with db.execute(
        """
        SELECT c.*, a.creator, a.source, a.title as asset_title
        FROM clips c JOIN assets a ON c.asset_id = a.id
        WHERE c.status IN ('rendered', 'candidate')
        ORDER BY c.overall_score DESC
        LIMIT 50
        """,
    ) as cur:
        rows = await cur.fetchall()
        cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


async def get_clip_by_id(db: aiosqlite.Connection, clip_id: str) -> Optional[Clip]:
    """Fetch a single clip by its primary ID."""
    async with db.execute("SELECT * FROM clips WHERE id=?", (clip_id,)) as cur:
        row = await cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        d = dict(zip(cols, row))
        return Clip(
            id=d["id"],
            asset_id=d["asset_id"],
            start_time=float(d["start_time"]),
            end_time=float(d["end_time"]),
            overall_score=float(d["overall_score"]),
            scores_breakdown=json.loads(d.get("scores_breakdown") or "{}"),
            raw_clip_path=Path(d["raw_clip_path"]) if d.get("raw_clip_path") else None,
            rendered_clip_path=Path(d["rendered_clip_path"]) if d.get("rendered_clip_path") else None,
            status=ClipStatus(d["status"]),
            title=d.get("title") or "",
            description=d.get("description") or "",
            tags=json.loads(d.get("tags_json") or "[]"),
            reviewed_at=datetime.fromisoformat(d["reviewed_at"]) if d.get("reviewed_at") else None,
            reviewed_by=d.get("reviewed_by"),
        )


async def get_approved_clips(db: aiosqlite.Connection, limit: int = 50) -> list[Clip]:
    """Fetch approved clips ready for social publishing."""
    async with db.execute(
        """
        SELECT * FROM clips
        WHERE status='approved' AND rendered_clip_path IS NOT NULL
        ORDER BY overall_score DESC
        LIMIT ?
        """,
        (limit,),
    ) as cur:
        rows = await cur.fetchall()
        cols = [d[0] for d in cur.description]

    clips = []
    for r in rows:
        d = dict(zip(cols, r))
        clips.append(
            Clip(
                id=d["id"],
                asset_id=d["asset_id"],
                start_time=float(d["start_time"]),
                end_time=float(d["end_time"]),
                overall_score=float(d["overall_score"]),
                scores_breakdown=json.loads(d.get("scores_breakdown") or "{}"),
                raw_clip_path=Path(d["raw_clip_path"]) if d.get("raw_clip_path") else None,
                rendered_clip_path=Path(d["rendered_clip_path"]) if d.get("rendered_clip_path") else None,
                status=ClipStatus(d["status"]),
                title=d.get("title") or "",
                description=d.get("description") or "",
                tags=json.loads(d.get("tags_json") or "[]"),
            )
        )
    return clips


async def set_clip_decision(
    db: aiosqlite.Connection,
    clip_id: str,
    decision: str,
    reason: str = "",
    in_delta: float = 0.0,
    out_delta: float = 0.0,
) -> None:
    new_status = "approved" if decision == "approved" else "rejected"
    await db.execute(
        "UPDATE clips SET status=?, reviewed_at=? WHERE id=?",
        (new_status, _now(), clip_id),
    )
    await db.execute(
        """
        INSERT INTO labels (clip_id, decision, reason, in_point_delta, out_point_delta)
        VALUES (?,?,?,?,?)
        """,
        (clip_id, decision, reason, in_delta, out_delta),
    )
    await db.commit()


# ── Post CRUD ─────────────────────────────────────────────────────────────────

async def insert_post(db: aiosqlite.Connection, post: Post) -> None:
    """Record a publication attempt in the posts table."""
    await db.execute(
        """
        INSERT INTO posts
            (id, clip_id, platform, account_id, remote_post_id, status,
             scheduled_for, published_at, view_count, like_count, error_message, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            post.id,
            post.clip_id,
            post.platform.value if hasattr(post.platform, "value") else str(post.platform),
            post.account_id,
            post.remote_post_id,
            post.status.value if hasattr(post.status, "value") else str(post.status),
            post.scheduled_for.isoformat() if post.scheduled_for else None,
            post.published_at.isoformat() if post.published_at else None,
            post.view_count,
            post.like_count,
            post.error_message,
            post.created_at.isoformat(),
        ),
    )
    await db.commit()


async def get_posts_for_clip(db: aiosqlite.Connection, clip_id: str) -> list[Post]:
    """Retrieve all posts associated with a specific clip."""
    async with db.execute("SELECT * FROM posts WHERE clip_id=?", (clip_id,)) as cur:
        rows = await cur.fetchall()
        cols = [d[0] for d in cur.description]

    posts = []
    for r in rows:
        d = dict(zip(cols, r))
        posts.append(
            Post(
                id=d["id"],
                clip_id=d["clip_id"],
                platform=Platform(d["platform"]),
                account_id=d["account_id"],
                remote_post_id=d["remote_post_id"],
                status=PostStatus(d["status"]),
                scheduled_for=datetime.fromisoformat(d["scheduled_for"]) if d.get("scheduled_for") else None,
                published_at=datetime.fromisoformat(d["published_at"]) if d.get("published_at") else None,
                view_count=d.get("view_count", 0),
                like_count=d.get("like_count", 0),
                error_message=d.get("error_message"),
            )
        )
    return posts
