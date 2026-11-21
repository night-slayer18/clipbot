"""Fast keyframe-accurate clip trimming via ffmpeg with micro-fade click prevention and 5.1 dialogue preservation."""

from __future__ import annotations

import asyncio
from pathlib import Path
import shutil

from loguru import logger


async def get_audio_channels(source: Path) -> int:
    """Detect number of audio channels in source video via ffprobe."""
    ffprobe_bin = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe_bin, "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(source),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0 and stdout.strip().isdigit():
            return int(stdout.strip())
    except Exception:
        pass
    return 2


async def cut_clip(
    source: Path,
    start: float,
    end: float,
    output: Path,
    fast: bool = True,
) -> Path:
    """
    Trim source video from start to end seconds.
    fast=True uses stream copy (lossless, instant).
    fast=False performs frame-accurate re-encode with 50ms cut-boundary micro-fades (eliminating audio clicks/pops)
    and 5.1 dialogue-preserving center-channel downmixing.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end - start)

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
        # Build audio filters: 40ms micro-fade to eliminate clicks, and 5.1 downmix if multichannel
        audio_filters = []
        channels = await get_audio_channels(source)
        if channels > 2:
            # Preserve dialogue from center channel and mix LFE subtly
            audio_filters.append("pan=stereo|FL=1.0*FL+1.0*FC+0.707*BL+0.3*LFE|FR=1.0*FR+1.0*FC+0.707*BR+0.3*LFE")

        # Ensure clean audio sync and resample to prevent drift
        audio_filters.append("aresample=async=1000:first_pts=0")

        fade_d = min(0.04, duration / 4)
        fade_st = max(0.0, duration - fade_d)
        audio_filters.append(f"afade=t=in:st=0:d={fade_d:.3f},afade=t=out:st={fade_st:.3f}:d={fade_d:.3f}")

        af_arg = ",".join(audio_filters)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(source),
            "-t", str(duration),
            "-fps_mode", "cfr",
            "-r", "60",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-af", af_arg,
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
