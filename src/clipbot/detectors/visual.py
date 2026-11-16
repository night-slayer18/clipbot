"""Universal visual detector — extracts scene transitions, kinetic motion, high-impact flashes, and intermission screens.

Signals produced:
  visual_scene        — 1Hz inter-frame scene cut / camera switch magnitude (universal)
  visual_motion       — 1Hz kinetic action intensity and optical displacement (sports, action, cartoons)
  visual_impact       — 1Hz high-contrast color flash / damage vignette / elimination screen (0.0 to 1.0)
  visual_intermission — 1Hz binary flag (1.0 = 'Starting Soon', BRB, static loading screen, or dark end credits)
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


def _is_dark_credit_frame(gray: np.ndarray) -> bool:
    """Detect if frame has dark background with high-contrast text edges (movie/cartoon credits)."""
    dark_ratio = float(np.mean(gray < 45))
    if dark_ratio < 0.70:
        return False
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.mean(edges > 0))
    return 0.003 < edge_ratio < 0.15


def _detect_intermissions_and_credits(frames_gray: list[np.ndarray], min_duration: int = 12) -> np.ndarray:
    """
    Detect static loading screens, 'Starting Soon' screens, BRB screens,
    and end credits sequences across the timeline.
    Returns 1Hz array where 1.0 = intermission / non-content screen.
    """
    n = len(frames_gray)
    is_non_content = np.zeros(n, dtype=bool)

    for i in range(n):
        gray = frames_gray[i]
        # 1. Dark credit check
        if _is_dark_credit_frame(gray):
            is_non_content[i] = True
            continue

        # 2. Static / frozen screen check against previous frame (loading screens, pause screens)
        if i > 0:
            diff = float(cv2.absdiff(gray, frames_gray[i - 1]).mean())
            if diff < 1.5:  # Nearly identical consecutive frame
                is_non_content[i] = True

    # Only flag runs that persist for at least min_duration consecutive seconds
    intermission_signal = np.zeros(n, dtype=np.float32)
    start_run = None
    for i in range(n):
        if is_non_content[i]:
            if start_run is None:
                start_run = i
        else:
            if start_run is not None:
                length = i - start_run
                if length >= min_duration:
                    intermission_signal[start_run:i] = 1.0
                start_run = None
    if start_run is not None and (n - start_run) >= min_duration:
        intermission_signal[start_run:n] = 1.0

    return intermission_signal


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
                "visual_intermission": np.zeros(dur, dtype=np.float32),
            }

        n = len(frames)
        raw_impact = np.zeros(n, dtype=np.float32)
        scene_signal = np.zeros(n, dtype=np.float32)
        motion_signal = np.zeros(n, dtype=np.float32)
        frames_gray: list[np.ndarray] = []

        prev_gray: np.ndarray | None = None
        for i, frame in enumerate(frames):
            raw_impact[i] = _detect_impact_vignette(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames_gray.append(gray)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                scene_signal[i] = float(diff.mean()) / 255.0
                motion_signal[i] = float(np.std(diff)) / 128.0
            prev_gray = gray

        # Detect sustained static loading screens, 'Starting Soon', BRB, and dark credits
        intermission_signal = _detect_intermissions_and_credits(frames_gray, min_duration=12)

        # Smooth impact signal to expand instantaneous hits to adjacent window (±2s)
        impact_smooth = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if raw_impact[i] > 0.5:
                lo = max(0, i - 2)
                hi = min(n, i + 3)
                impact_smooth[lo:hi] = np.maximum(impact_smooth[lo:hi], raw_impact[i])

        logger.debug(
            f"[Visual] {asset.external_id}: {n}s, impact peaks={int((impact_smooth > 0.5).sum())}, intermission={int(intermission_signal.sum())}s"
        )
        return {
            "visual_impact": impact_smooth,
            "visual_scene": scene_signal,
            "visual_motion": motion_signal,
            "visual_intermission": intermission_signal,
        }

