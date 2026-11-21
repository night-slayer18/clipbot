"""Media inspection utilities (ffprobe duration and video metadata extraction)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from loguru import logger


def probe_media_duration(path: Path) -> float:
    """Accurately probe media file duration in seconds using ffprobe with OpenCV fallback."""
    if not path.exists():
        return 0.0

    ffprobe_bin = shutil.which("ffprobe") or "ffprobe"
    try:
        cmd = [
            ffprobe_bin,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=10.0).decode().strip()
        dur = float(out)
        if dur > 0:
            return dur
    except Exception as exc:
        logger.debug(f"[Media] ffprobe duration probe notice for {path.name}: {exc}")

    # Fallback to OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            if frames > 0 and fps > 0:
                return float(frames / fps)
    except Exception:
        pass

    return 0.0


def probe_media_resolution(path: Path) -> tuple[int, int]:
    """Return (width, height) of video file."""
    ffprobe_bin = shutil.which("ffprobe") or "ffprobe"
    try:
        cmd = [
            ffprobe_bin,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=10.0).decode().strip()
        data = json.loads(out)
        stream = data.get("streams", [{}])[0]
        return int(stream.get("width", 1920)), int(stream.get("height", 1080))
    except Exception:
        return 1920, 1080
