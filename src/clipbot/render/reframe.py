"""9:16 vertical reframing for GTA highlights."""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger


async def reframe_split_stack(
    source: Path,
    output: Path,
    top_crop: str = "1280:720:0:0",    # facecam region (w:h:x:y)
    bottom_crop: str = "1280:720:0:0", # gameplay region (w:h:x:y)
    canvas_width: int = 1080,
    canvas_height: int = 1920,
) -> Path:
    """
    Stack facecam (top ~35%) above gameplay (bottom ~65%) in 9:16.
    Uses ffmpeg complex filtergraph.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    top_h = int(canvas_height * 0.35)
    bottom_h = canvas_height - top_h

    # FFmpeg filtergraph:
    # 1. Crop facecam region, scale to canvas_width x top_h
    # 2. Crop gameplay region, scale to canvas_width x bottom_h
    # 3. Stack vertically
    filtergraph = (
        f"[0:v]crop={top_crop},scale={canvas_width}:{top_h},setsar=1[top];"
        f"[0:v]crop={bottom_crop},scale={canvas_width}:{bottom_h},setsar=1[bottom];"
        f"[top][bottom]vstack=inputs=2[out]"
    )
    cmd = [
        "ffmpeg", "-y", "-i", str(source),
        "-filter_complex", filtergraph,
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output),
    ]
    logger.info(f"[Reframe] Split-stack → {output.name}")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg reframe failed: {stderr.decode()[-300:]}")
    return output


async def reframe_blur_canvas(
    source: Path,
    output: Path,
    canvas_width: int = 1080,
    canvas_height: int = 1920,
) -> Path:
    """
    Center original 16:9 video on a 9:16 blurred background canvas.
    Best when no facecam is present.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    # Scale source to fit canvas width, blur a background, overlay sharp foreground
    video_h = int(canvas_width * 9 / 16)
    y_offset = (canvas_height - video_h) // 2

    filtergraph = (
        f"[0:v]scale=-2:{canvas_height},crop={canvas_width}:{canvas_height},"
        f"gblur=sigma=22[bg];"
        f"[0:v]scale={canvas_width}:{video_h}[fg];"
        f"[bg][fg]overlay=0:{y_offset}[out]"
    )
    cmd = [
        "ffmpeg", "-y", "-i", str(source),
        "-filter_complex", filtergraph,
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output),
    ]
    logger.info(f"[Reframe] Blur canvas → {output.name}")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg blur reframe failed: {stderr.decode()[-300:]}")
    return output
