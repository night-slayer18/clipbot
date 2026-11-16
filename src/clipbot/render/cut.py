"""Fast keyframe-accurate clip trimming via ffmpeg."""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger


async def cut_clip(
    source: Path,
    start: float,
    end: float,
    output: Path,
    fast: bool = True,
) -> Path:
    """
    Trim source video from start to end seconds.
    fast=True uses stream copy (lossless, instant). Set False for frame-accurate re-encode.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start

    if fast:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(source),
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(source),
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            str(output),
        ]

    logger.info(f"[Cut] {source.name} [{start:.1f}s → {end:.1f}s] → {output.name}")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed: {stderr.decode()[-300:]}")
    return output
