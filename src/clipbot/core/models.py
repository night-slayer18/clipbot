"""Core dataclasses and enums for ClipBot."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AssetStatus(str, Enum):
    DISCOVERED = "discovered"
    DOWNLOADING = "downloading"
    READY = "ready"
    SCORED = "scored"
    ARCHIVED = "archived"
    ERROR = "error"


class ClipStatus(str, Enum):
    CANDIDATE = "candidate"
    RENDERED = "rendered"
    APPROVED = "approved"
    REJECTED = "rejected"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"


class PostStatus(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    UPLOADING = "uploading"
    PUBLISHED = "published"
    FAILED = "failed"


class Platform(str, Enum):
    YOUTUBE_SHORTS = "youtube_shorts"
    TIKTOK = "tiktok"
    INSTAGRAM_REELS = "instagram_reels"


class Source(str, Enum):
    TWITCH = "twitch"
    KICK = "kick"
    YOUTUBE = "youtube"


@dataclass
class Asset:
    """Represents a source video asset (clip or VOD segment from another creator)."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: Source = Source.TWITCH
    external_id: str = ""
    creator: str = ""
    title: str = ""
    game: str = "General"
    url: str = ""
    duration: Optional[float] = None           # seconds
    local_path: Optional[Path] = None          # downloaded media
    chat_path: Optional[Path] = None           # chat log jsonl
    status: AssetStatus = AssetStatus.DISCOVERED
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)

    @property
    def is_ready(self) -> bool:
        return self.local_path is not None and self.local_path.exists()


@dataclass
class SignalSeries:
    """Per-second time-series signal from a detector."""

    asset_id: str
    detector: str
    # numpy array serialized as bytes; shape=(duration_seconds,) dtype=float32
    series_bytes: bytes
    created_at: datetime = field(default_factory=_utcnow)


@dataclass
class Clip:
    """A candidate or rendered highlight clip."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    asset_id: str = ""
    start_time: float = 0.0            # seconds into the source asset
    end_time: float = 0.0
    overall_score: float = 0.0
    scores_breakdown: dict = field(default_factory=dict)
    raw_clip_path: Optional[Path] = None
    rendered_clip_path: Optional[Path] = None  # Final 9:16 short
    status: ClipStatus = ClipStatus.CANDIDATE
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    created_at: datetime = field(default_factory=_utcnow)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class ReviewLabel:
    """Human feedback on a clip (drives adaptive scoring)."""

    clip_id: str
    decision: str                     # 'approved' or 'rejected'
    reason: Optional[str] = None
    in_point_delta: float = 0.0       # manual start adjustment (seconds)
    out_point_delta: float = 0.0
    created_at: datetime = field(default_factory=_utcnow)


@dataclass
class Post:
    """A scheduled or published post to a social platform."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    clip_id: str = ""
    platform: Platform = Platform.YOUTUBE_SHORTS
    account_id: str = "default"
    remote_post_id: Optional[str] = None
    status: PostStatus = PostStatus.PENDING
    scheduled_for: Optional[datetime] = None
    published_at: Optional[datetime] = None
    view_count: int = 0
    like_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=_utcnow)
