"""Pytest fixtures and real-media synthetic generators for zero-mock testing."""

from __future__ import annotations

import math
import struct
import tempfile
import wave
from pathlib import Path
from typing import AsyncGenerator

import aiosqlite
import cv2
import numpy as np
import pytest

from clipbot.core.db import SCHEMA_SQL
from clipbot.core.models import Asset, AssetStatus, Clip, ClipStatus, Post, PostStatus, Platform, Source


def generate_real_wav(
    dest_path: Path,
    duration_sec: float = 3.0,
    sample_rate: int = 16000,
    freq: float = 440.0,
    with_silence: bool = False,
) -> Path:
    """Generate a real valid 16kHz mono WAV file with sine wave audio and optional silence."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    
    with wave.open(str(dest_path), "wb") as wav:
        wav.setnchannels(1)        # mono
        wav.setsampwidth(2)        # 16-bit
        wav.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(num_samples):
            t = float(i) / sample_rate
            # If silence requested, mute middle section
            if with_silence and (duration_sec * 0.3 < t < duration_sec * 0.7):
                val = 0
            else:
                val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * freq * t))
            frames.extend(struct.pack("<h", val))
            
        wav.writeframes(frames)
    return dest_path


def generate_real_mp4(
    dest_path: Path,
    duration_sec: int = 3,
    fps: int = 30,
    width: int = 640,
    height: int = 360,
    color: tuple[int, int, int] = (0, 0, 255),  # BGR
) -> Path:
    """Generate a real valid MP4 video using OpenCV."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dest_path), fourcc, fps, (width, height))
    
    total_frames = int(duration_sec * fps)
    for i in range(total_frames):
        # Create solid color frame with a moving rectangle to simulate motion
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        rect_x = int((i / total_frames) * (width - 60))
        cv2.rectangle(frame, (rect_x, height // 3), (rect_x + 60, height // 3 + 60), (255, 255, 255), -1)
        out.write(frame)
        
    out.release()
    return dest_path


@pytest.fixture
async def test_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Provide a real in-memory async SQLite database with schema and foreign keys enabled."""
    db = await aiosqlite.connect(":memory:")
    await db.execute("PRAGMA foreign_keys = ON;")
    await db.executescript(SCHEMA_SQL)
    await db.commit()
    yield db
    await db.close()


@pytest.fixture
def sample_asset() -> Asset:
    return Asset(
        id="test-asset-1",
        source=Source.TWITCH,
        external_id="ext-12345",
        creator="Ninja",
        title="Epic GTA 6 Chaos in Vice City",
        game="Grand Theft Auto VI",
        url="https://twitch.tv/ninja/clip/12345",
        duration=45.0,
        status=AssetStatus.READY,
    )


@pytest.fixture
def sample_clip(sample_asset: Asset) -> Clip:
    return Clip(
        id="clip-abc-123",
        asset_id=sample_asset.id,
        start_time=5.0,
        end_time=35.0,
        overall_score=0.92,
        scores_breakdown={"audio_rms": 0.85, "chat_velocity": 0.95},
        status=ClipStatus.APPROVED,
        title="🔥 Insane Escape in Vice City!",
        description="Watch this wild moment #shorts",
        tags=["gta6", "gaming", "shorts"],
    )
