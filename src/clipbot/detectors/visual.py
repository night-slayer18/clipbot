"""Universal visual detector — extracts scene transitions, kinetic motion, and high-impact visual flashes.

Signals produced:
  visual_scene   — 1Hz inter-frame scene cut / camera switch magnitude (universal)
  visual_motion  — 1Hz kinetic action intensity and optical displacement (sports, action, cartoons)
  visual_impact  — 1Hz high-contrast color flash / damage vignette / elimination screen (0.0 to 1.0)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from clipbot.core.models import Asset

# High-impact color vignette / flash thresholds
SAT_UPPER_BOUND = 220    # Max saturation threshold to avoid over-saturated graphics
RED_RATIO_THRESH = 0.45  # R / (R+G+B) indicating heavy red damage/impact flash
BLUE_RATIO_THRESH = 0.40 # B / (R+G+B) indicating heavy blue flash/freeze
SAMPLE_FPS = 1          # Sample 1 frame per second for efficient time-series extraction


def _detect_impact_vignette(frame_bgr: np.ndarray) -> float:
    """Return score 0.0–1.0 indicating high-contrast impact flash or damage vignette."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())
    if mean_sat > SAT_UPPER_BOUND:
        return 0.0

    b = float(frame_bgr[:, :, 0].mean())
    g = float(frame_bgr[:, :, 1].mean())
    r = float(frame_bgr[:, :, 2].mean())
    total = b + g + r + 1e-6
    red_ratio = r / total
    blue_ratio = b / total

    if red_ratio > RED_RATIO_THRESH:
        return 1.0
    if blue_ratio > BLUE_RATIO_THRESH:
        return 0.85
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
        """Run visual analysis on media path producing 1Hz time-series signals."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._analyse_sync, media_path, asset
        )

    def _analyse_sync(self, media_path: Path, asset: Asset) -> dict[str, np.ndarray]:
        logger.debug(f"[Visual] Analysing {media_path.name}")
        try:
            frames = _extract_frames_1fps(media_path, SAMPLE_FPS)
        except Exception as exc:
            logger.warning(f"[Visual] Frame extraction failed: {exc}")
            dur = int(asset.duration or 60)
            return {
                "visual_impact": np.zeros(dur, dtype=np.float32),
                "visual_scene": np.zeros(dur, dtype=np.float32),
                "visual_motion": np.zeros(dur, dtype=np.float32),
            }

        n = len(frames)
        raw_impact = np.zeros(n, dtype=np.float32)
        scene_signal = np.zeros(n, dtype=np.float32)
        motion_signal = np.zeros(n, dtype=np.float32)

        prev_gray: np.ndarray | None = None
        for i, frame in enumerate(frames):
            raw_impact[i] = _detect_impact_vignette(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                scene_signal[i] = float(diff.mean()) / 255.0
                motion_signal[i] = float(np.std(diff)) / 128.0
            prev_gray = gray

        # Smooth impact signal to expand instantaneous hits to adjacent window (±2s)
        impact_smooth = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if raw_impact[i] > 0.5:
                lo = max(0, i - 2)
                hi = min(n, i + 3)
                impact_smooth[lo:hi] = np.maximum(impact_smooth[lo:hi], raw_impact[i])

        logger.debug(
            f"[Visual] {asset.external_id}: {n}s, impact peaks={int((impact_smooth > 0.5).sum())}"
        )
        return {
            "visual_impact": impact_smooth,
            "visual_scene": scene_signal,
            "visual_motion": motion_signal,
        }
