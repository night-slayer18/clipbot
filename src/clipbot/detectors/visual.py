"""Universal visual detector — extracts scene transitions, kinetic motion, high-impact flashes, and intermission screens.

Streaming execution: Processes frames sequentially at 1 fps with O(1) memory overhead.
Never accumulates raw frames in memory, eliminating OOM crashes on multi-hour streams.

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

from clipbot.core.media import probe_media_duration
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


class VisualDetector:
    name = "visual"

    async def signals(self, asset: Asset, media_path: Path) -> dict[str, np.ndarray]:
        """Run visual analysis on media path producing 1Hz time-series signals."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._analyse_sync, media_path, asset
        )

    def _analyse_sync(self, media_path: Path, asset: Asset) -> dict[str, np.ndarray]:
        logger.debug(f"[Visual] Analysing {media_path.name}")
        cache_file = media_path.with_name(f"{media_path.stem}_visual_signals.npz")
        if cache_file.exists():
            try:
                data = np.load(str(cache_file))
                logger.info(f"[Visual] Loaded cached signals from {cache_file.name}")
                return {k: data[k] for k in data.files}
            except Exception as e:
                logger.warning(f"[Visual] Failed loading cache {cache_file}: {e}")

        cap = cv2.VideoCapture(str(media_path))
        if not cap.isOpened():
            logger.warning(f"[Visual] Cannot open video file: {media_path}")
            dur = int(probe_media_duration(media_path) or asset.duration or 60)
            return {
                "visual_impact": np.zeros(dur, dtype=np.float32),
                "visual_scene": np.zeros(dur, dtype=np.float32),
                "visual_motion": np.zeros(dur, dtype=np.float32),
                "visual_intermission": np.zeros(dur, dtype=np.float32),
            }

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(native_fps / SAMPLE_FPS)))

        raw_impact_list: list[float] = []
        scene_list: list[float] = []
        motion_list: list[float] = []
        is_non_content_list: list[bool] = []

        prev_gray: np.ndarray | None = None
        frame_idx = 0

        # Stream frames sequentially: O(1) memory usage regardless of video duration
        while True:
            if frame_idx % step == 0:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Downsample frame to 360p height for fast metric computation
                h, w = frame.shape[:2]
                if h > 360:
                    scale = 360.0 / h
                    frame_small = cv2.resize(frame, (int(w * scale), 360), interpolation=cv2.INTER_AREA)
                else:
                    frame_small = frame

                # 1. Color flash / damage vignette
                impact_val = _detect_impact_vignette(frame_small)
                raw_impact_list.append(impact_val)

                # 2. Convert to grayscale for scene/motion/intermission checks
                gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                is_credit = _is_dark_credit_frame(gray)
                is_static = False

                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    diff_mean = float(diff.mean())
                    scene_list.append(diff_mean / 255.0)
                    motion_list.append(float(np.std(diff)) / 128.0)
                    if diff_mean < 1.5:
                        is_static = True
                else:
                    scene_list.append(0.0)
                    motion_list.append(0.0)

                is_non_content_list.append(bool(is_credit or is_static))
                prev_gray = gray
                # Frame memory released immediately
            else:
                ret = cap.grab()
                if not ret:
                    break
            frame_idx += 1

        cap.release()

        n = len(raw_impact_list)
        if n == 0:
            dur = int(probe_media_duration(media_path) or asset.duration or 60)
            return {
                "visual_impact": np.zeros(dur, dtype=np.float32),
                "visual_scene": np.zeros(dur, dtype=np.float32),
                "visual_motion": np.zeros(dur, dtype=np.float32),
                "visual_intermission": np.zeros(dur, dtype=np.float32),
            }

        raw_impact = np.array(raw_impact_list, dtype=np.float32)
        scene_signal = np.array(scene_list, dtype=np.float32)
        motion_signal = np.array(motion_list, dtype=np.float32)

        # Detect sustained static loading screens, 'Starting Soon', BRB, and dark credits (min 12s)
        intermission_signal = np.zeros(n, dtype=np.float32)
        start_run = None
        for i, non_content in enumerate(is_non_content_list):
            if non_content:
                if start_run is None:
                    start_run = i
            else:
                if start_run is not None:
                    if (i - start_run) >= 12:
                        intermission_signal[start_run:i] = 1.0
                    start_run = None
        if start_run is not None and (n - start_run) >= 12:
            intermission_signal[start_run:n] = 1.0

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
        res = {
            "visual_impact": impact_smooth,
            "visual_scene": scene_signal,
            "visual_motion": motion_signal,
            "visual_intermission": intermission_signal,
        }
        try:
            np.savez_compressed(str(cache_file), **res)
        except Exception as e:
            logger.debug(f"[Visual] Failed saving cache {cache_file}: {e}")
        return res
