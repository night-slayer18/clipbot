"""Universal visual detector — extracts scene transitions, kinetic motion, high-impact flashes, and intermission screens.

Streaming execution: Processes one frame per second via millisecond seeking with O(1) memory overhead.
Never accumulates raw frames in memory, eliminating OOM crashes on multi-hour streams.

Signals produced:
  visual_scene        — 1Hz inter-frame scene cut / camera switch magnitude (universal)
  visual_motion       — 1Hz kinetic action intensity and optical displacement (sports, action, cartoons)
  visual_impact       — 1Hz high-contrast color flash / damage vignette / elimination screen (0.0 to 1.0)
  visual_intermission — 1Hz binary flag (1.0 = 'Starting Soon', BRB, static loading screen, or dark end credits)
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from clipbot.core.media import probe_media_duration
from clipbot.core.models import Asset

SAT_UPPER_BOUND = 220
RED_RATIO_THRESH = 0.45
BLUE_RATIO_THRESH = 0.40
SAMPLE_FPS = 1

_VISUAL_CACHE_VERSION = "v2"


def _visual_cache_key(media_path: Path) -> str:
    path_hash = hashlib.sha1(str(media_path).encode()).hexdigest()[:12]
    return f"{media_path.stem}_{path_hash}_visual_signals_{_VISUAL_CACHE_VERSION}.npz"


def _probe_fps_ffprobe(media_path: Path) -> float:
    ffprobe_bin = shutil.which("ffprobe") or "ffprobe"
    try:
        result = subprocess.run(
            [
                ffprobe_bin, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            fps_str = result.stdout.strip()
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            if 1.0 <= fps <= 240.0:
                return fps
    except Exception as e:
        logger.debug(f"[Visual] ffprobe fps probe failed: {e}")
    return 30.0


def _detect_impact_vignette(frame_bgr: np.ndarray, diff_mean: float = 0.0) -> float:
    """Return score 0.0–1.0 indicating high-contrast impact flash or damage vignette with motion."""
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

    has_motion = diff_mean > 2.0

    if red_ratio > RED_RATIO_THRESH and has_motion:
        return 1.0
    if blue_ratio > BLUE_RATIO_THRESH and has_motion:
        return 0.85
    return 0.0


def _is_dark_credit_frame(gray: np.ndarray) -> bool:
    dark_ratio = float(np.mean(gray < 45))
    if dark_ratio < 0.70:
        return False
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.mean(edges > 0))
    return 0.003 < edge_ratio < 0.15


class VisualDetector:
    name = "visual"

    async def signals(self, asset: Asset, media_path: Path) -> dict[str, np.ndarray]:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._analyse_sync, media_path, asset
        )

    def _analyse_sync(self, media_path: Path, asset: Asset) -> dict[str, np.ndarray]:
        logger.debug(f"[Visual] Analysing {media_path.name}")

        cache_name = _visual_cache_key(media_path)
        cache_file = media_path.parent / cache_name
        if cache_file.exists():
            try:
                data = np.load(str(cache_file))
                logger.info(f"[Visual] Loaded cached signals from {cache_file.name}")
                return {k: data[k] for k in data.files}
            except Exception as e:
                logger.warning(f"[Visual] Failed loading cache {cache_file}: {e}")

        native_fps = _probe_fps_ffprobe(media_path)

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

        raw_impact_list: list[float] = []
        scene_list: list[float] = []
        motion_list: list[float] = []
        is_non_content_list: list[bool] = []

        prev_gray: np.ndarray | None = None
        t_ms = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t_ms))
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            h, w = frame.shape[:2]
            if h > 360:
                scale = 360.0 / h
                frame_small = cv2.resize(frame, (int(w * scale), 360), interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame

            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            is_credit = _is_dark_credit_frame(gray)
            is_static = False
            diff_mean = 0.0

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

            impact_val = _detect_impact_vignette(frame_small, diff_mean=diff_mean)
            raw_impact_list.append(impact_val)

            is_non_content_list.append(bool(is_credit or is_static))
            prev_gray = gray

            t_ms += 1000

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

        impact_smooth = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if raw_impact[i] > 0.5:
                lo = max(0, i - 2)
                hi = min(n, i + 3)
                impact_smooth[lo:hi] = np.maximum(impact_smooth[lo:hi], raw_impact[i])

        logger.debug(
            f"[Visual] {asset.external_id}: {n}s, impact peaks={int((impact_smooth > 0.5).sum())}, "
            f"intermission={int(intermission_signal.sum())}s, fps_used={native_fps:.2f}"
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
