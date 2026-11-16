"""Async SQLite database layer with schema migrations."""

from __future__ import annotations

import aiosqlite
from pathlib import Path
from loguru import logger


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    creator TEXT NOT NULL,
    title TEXT,
    game TEXT DEFAULT 'General',
    url TEXT NOT NULL,
    duration REAL,
    local_path TEXT,
    chat_path TEXT,
    status TEXT NOT NULL DEFAULT 'discovered',
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source, external_id)
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    detector TEXT NOT NULL,
    series_blob BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(asset_id, detector)
);

CREATE TABLE IF NOT EXISTS clips (
    id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    overall_score REAL NOT NULL,
    scores_breakdown TEXT,
    raw_clip_path TEXT,
    rendered_clip_path TEXT,
    status TEXT NOT NULL DEFAULT 'candidate',
    title TEXT,
    description TEXT,
    tags_json TEXT,
    reviewed_at TIMESTAMP,
    reviewed_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    decision TEXT NOT NULL,
    reason TEXT,
    in_point_delta REAL DEFAULT 0.0,
    out_point_delta REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    clip_id TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    platform TEXT NOT NULL,
    account_id TEXT DEFAULT 'default',
    remote_post_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    scheduled_for TIMESTAMP,
    published_at TIMESTAMP,
    view_count INTEGER DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_assets_status ON assets(status);
CREATE INDEX IF NOT EXISTS idx_clips_status ON clips(status);
CREATE INDEX IF NOT EXISTS idx_clips_asset ON clips(asset_id);
CREATE INDEX IF NOT EXISTS idx_posts_platform_status ON posts(platform, status);
"""


async def init_db(db_path: str = "data/state.db") -> None:
    """Initialize the database and create schema tables."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path, timeout=30.0) as db:
        await db.execute("PRAGMA journal_mode = WAL;")
        await db.execute("PRAGMA foreign_keys = ON;")
        await db.execute("PRAGMA busy_timeout = 10000;")
        await db.executescript(SCHEMA_SQL)
        await db.commit()
    logger.info(f"Database initialized at {db_path} (WAL mode enabled)")


async def get_db(db_path: str = "data/state.db") -> aiosqlite.Connection:
    """Get an open async DB connection (caller must close)."""
    db = await aiosqlite.connect(db_path, timeout=30.0)
    await db.execute("PRAGMA foreign_keys = ON;")
    await db.execute("PRAGMA busy_timeout = 10000;")
    return db
