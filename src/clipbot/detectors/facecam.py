"""Facecam detector using OpenCV Haar Cascades.

Detects streamer facecam coordinates (x, y, w, h) across sample frames.
If a consistent face is detected in a quadrant, returns the facecam bounding box.
If no face is detected (gameplay-only stream), returns None so the blur canvas layout is used.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger


@dataclass
class FacecamBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float

    def to_ffmpeg_crop(self) -> str:
        """Format as FFmpeg crop parameter: w:h:x:y (ensuring even dimensions for yuv420p)."""
        even_w = (self.w // 2) * 2
        even_h = (self.h // 2) * 2
        return f"{even_w}:{even_h}:{self.x}:{self.y}"


class FacecamDetector:
    """Detects streamer facecam regions in 16:9 gameplay streams."""

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)

    async def detect(self, video_path: Path, sample_count: int = 10) -> Optional[FacecamBox]:
        """Async wrapper to detect facecam box across sampled frames."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._detect_sync, video_path, sample_count
        )

    def _detect_sync(self, video_path: Path, sample_count: int = 10) -> Optional[FacecamBox]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"[Facecam] Cannot open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            cap.release()
            return None

        step = max(30, total_frames // (sample_count + 1)) if total_frames > 0 else 60
        detected_faces = []

        for i in range(1, sample_count + 1):
            frame_idx = i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(int(height * 0.08), int(height * 0.08)),
                maxSize=(int(height * 0.5), int(height * 0.5)),
            )

            for (x, y, w, h) in faces:
                detected_faces.append((x, y, w, h))

        cap.release()

        if not detected_faces:
            logger.debug(f"[Facecam] No face detected in {video_path.name}")
            return None

        # Cluster / median face location
        xs = [f[0] for f in detected_faces]
        ys = [f[1] for f in detected_faces]
        ws = [f[2] for f in detected_faces]
        hs = [f[3] for f in detected_faces]

        median_x = int(np.median(xs))
        median_y = int(np.median(ys))
        median_w = int(np.median(ws))
        median_h = int(np.median(hs))

        # Pad around the face to capture the full webcam frame (head + shoulders + room)
        pad_x = int(median_w * 0.6)
        pad_y = int(median_h * 0.6)

        cam_x = max(0, median_x - pad_x)
        cam_y = max(0, median_y - pad_y)
        cam_w = min(width - cam_x, median_w + 2 * pad_x)
        cam_h = min(height - cam_y, median_h + 2 * pad_y)

        # Ensure 16:9 or roughly horizontal aspect ratio for top slot
        target_cam_h = int(cam_w * 9 / 16)
        if cam_y + target_cam_h <= height:
            cam_h = target_cam_h

        confidence = len(detected_faces) / sample_count
        logger.info(f"[Facecam] Detected facecam: {cam_w}x{cam_h} at ({cam_x},{cam_y}), confidence={confidence:.2f}")

        return FacecamBox(x=cam_x, y=cam_y, w=cam_w, h=cam_h, confidence=confidence)
