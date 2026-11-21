"""9:16 vertical reframing for stream and video highlights."""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger


async def reframe_split_stack(
    source: Path,
    output: Path,
    top_crop: str = "1280:720:0:0",    # facecam region (w:h:x:y)
    bottom_crop: str = "934:1080:493:0", # gameplay region (w:h:x:y, matching 1080:1248 aspect ratio)
    canvas_width: int = 1080,
    canvas_height: int = 1920,
) -> Path:
    """
    Stack facecam (top ~35%) above gameplay (bottom ~65%) in 9:16.
    Ensures non-distorted aspect ratio and broadcast-grade visual clarity (CRF 18, Lanczos).
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    top_h = int(canvas_height * 0.35)  # 672
    bottom_h = canvas_height - top_h   # 1248

    # FFmpeg filtergraph:
    # 1. Top facecam bg: scale with increase so it completely fills top_w x top_h, blurred
    # 2. Top facecam fg: scale with decrease so full facecam fits without distortion
    # 3. Bottom gameplay: crop matching exact aspect ratio (1080:1248), scaled to bottom slot
    filtergraph = (
        f"[0:v]crop={top_crop},scale={canvas_width}:{top_h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={canvas_width}:{top_h},gblur=sigma=20[top_bg];"
        f"[0:v]crop={top_crop},scale={canvas_width}:{top_h}:force_original_aspect_ratio=decrease:flags=lanczos[top_fg];"
        f"[top_bg][top_fg]overlay=(W-w)/2:(H-h)/2,setsar=1[top];"
        f"[0:v]crop={bottom_crop},scale={canvas_width}:{bottom_h}:flags=lanczos,setsar=1[bottom];"
        f"[top][bottom]vstack=inputs=2[out]"
    )
    cmd = [
        "ffmpeg", "-y", "-i", str(source),
        "-filter_complex", filtergraph,
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-b:v", "8M", "-maxrate", "12M", "-bufsize", "16M",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output),
    ]
    logger.info(f"[Reframe] Split-stack → {output.name} (CRF 18 High Quality)")
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
    Best when no facecam is present — 100% full-resolution, non-destructive clarity.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    video_h = int(canvas_width * 9 / 16)
    y_offset = (canvas_height - video_h) // 2

    filtergraph = (
        f"[0:v]scale={canvas_width}:{canvas_height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={canvas_width}:{canvas_height},"
        f"gblur=sigma=24[bg];"
        f"[0:v]scale={canvas_width}:{video_h}:flags=lanczos[fg];"
        f"[bg][fg]overlay=0:{y_offset},setsar=1[out]"
    )
    cmd = [
        "ffmpeg", "-y", "-i", str(source),
        "-filter_complex", filtergraph,
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-b:v", "8M", "-maxrate", "12M", "-bufsize", "16M",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output),
    ]
    logger.info(f"[Reframe] Blur canvas → {output.name} (CRF 18 High Quality)")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg blur reframe failed: {stderr.decode()[-300:]}")
    return output
