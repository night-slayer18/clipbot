"""Visual detector — detects GTA-specific screens using OpenCV.

Signals produced:
  visual_wasted  — 1.0 at seconds containing a WASTED/BUSTED screen (desaturated red overlay)
  visual_scene   — scene-change magnitude (1Hz), useful as a secondary chaos signal
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from clipbot.core.models import Asset


# ── WASTED / BUSTED detection ────────────────────────────────────────────────
# GTA V WASTED screen:  frame goes strongly desaturated + red-channel dominant
#   hsv_s < DESAT_THRESH  AND  red_ratio > RED_RATIO_THRESH
# BUSTED screen is similar (blue-dominant) so we catch it with the desaturation
# alone and then trust audio / chat to confirm.
DESAT_THRESH = 180      # saturation channel threshold (0-255)
RED_RATIO_THRESH = 0.45  # R / (R+G+B) across full frame
BLUE_RATIO_THRESH = 0.40  # B / (R+G+B) for BUSTED
SAMPLE_FPS = 1          # sample one frame per second (fast enough, cheap)


def _frame_is_wasted(frame_bgr: np.ndarray) -> float:
    """Return score 0–1.0 indicating likelihood of WASTED/BUSTED screen."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())
    if mean_sat > DESAT_THRESH:
        return 0.0  # Colour-rich frame — not a death screen

    # Check channel dominance (note: OpenCV is BGR)
    b = float(frame_bgr[:, :, 0].mean())
    g = float(frame_bgr[:, :, 1].mean())
    r = float(frame_bgr[:, :, 2].mean())
    total = b + g + r + 1e-6
    red_ratio = r / total
    blue_ratio = b / total

    if red_ratio > RED_RATIO_THRESH:
        return 1.0   # WASTED
    if blue_ratio > BLUE_RATIO_THRESH:
        return 0.85  # BUSTED (slightly lower confidence)
    return 0.0


def _extract_frames_1fps(video_path: Path, sample_fps: int = 1) -> list[np.ndarray]:
    """Extract one frame per second from a video file using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(native_fps / sample_fps))

    frames: list[np.ndarray] = []
    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_idx += step
        if total_frames > 0 and frame_idx >= total_frames:
            break

    cap.release()
    return frames


class VisualDetector:
    name = "visual"

    async def signals(self, asset: Asset, media_path: Path) -> dict[str, np.ndarray]:
        """
        Run visual analysis on the downloaded clip/VOD.
        Returns:
          visual_wasted  — 1Hz float array, 1.0 = WASTED/BUSTED screen detected
          visual_scene   — 1Hz scene-change magnitude
        """
        # Run blocking OpenCV work in a thread to keep asyncio happy
        result = await asyncio.get_event_loop().run_in_executor(
            None, self._analyse_sync, media_path, asset
        )
        return result

    def _analyse_sync(self, media_path: Path, asset: Asset) -> dict[str, np.ndarray]:
        logger.debug(f"[Visual] Analysing {media_path.name}")
        try:
            frames = _extract_frames_1fps(media_path, SAMPLE_FPS)
        except Exception as exc:
            logger.warning(f"[Visual] Frame extraction failed: {exc}")
            dur = int(asset.duration or 60)
            return {
                "visual_wasted": np.zeros(dur, dtype=np.float32),
                "visual_scene": np.zeros(dur, dtype=np.float32),
            }

        n = len(frames)
        wasted_signal = np.zeros(n, dtype=np.float32)
        scene_signal = np.zeros(n, dtype=np.float32)

        prev_gray: np.ndarray | None = None
        for i, frame in enumerate(frames):
            # 1. WASTED detection
            wasted_signal[i] = _frame_is_wasted(frame)

            # 2. Scene-change magnitude (absolute mean diff of consecutive gray frames)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = float(cv2.absdiff(gray, prev_gray).mean())
                scene_signal[i] = diff / 255.0  # normalise to 0-1 range
            prev_gray = gray

        # Smooth wasted signal: expand 1 second of WASTED to ±2s window (pre/post-death)
        wasted_smooth = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if wasted_signal[i] > 0.5:
                lo = max(0, i - 2)
                hi = min(n, i + 3)
                wasted_smooth[lo:hi] = np.maximum(wasted_smooth[lo:hi], wasted_signal[i])

        logger.debug(
            f"[Visual] {asset.external_id}: {n}s, wasted peaks={int((wasted_smooth > 0.5).sum())}"
        )
        return {"visual_wasted": wasted_smooth, "visual_scene": scene_signal}
