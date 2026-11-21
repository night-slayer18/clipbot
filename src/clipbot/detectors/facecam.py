"""Facecam detector using OpenCV Haar Cascades with corner-quadrant clamping and temporal stability.

Detects streamer facecam coordinates (x, y, w, h) across sample frames.
Enforces corner-quadrant boundaries to reject in-game character faces in the central gameplay area.
Clusters detections temporally across frames to verify stationary webcam boxes.
If confidence is insufficient, returns None so the non-destructive blur canvas layout is used.
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
    quadrant: str = "unknown"

    def to_ffmpeg_crop(self, max_w: int = 1920, max_h: int = 1080) -> str:
        """Format as FFmpeg crop parameter: w:h:x:y (ensuring even dimensions for yuv420p)."""
        safe_x = max(0, min(self.x, max_w - 2))
        safe_y = max(0, min(self.y, max_h - 2))
        safe_w = min(self.w, max_w - safe_x)
        safe_h = min(self.h, max_h - safe_y)
        even_w = max(2, (safe_w // 2) * 2)
        even_h = max(2, (safe_h // 2) * 2)
        return f"{even_w}:{even_h}:{safe_x}:{safe_y}"


def classify_face_quadrant(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> Optional[str]:
    """
    Classify which screen corner quadrant a face resides in.
    Returns None if the face falls outside the 4 extreme outer corners.
    Streamers place webcams strictly in the top-left, top-right, bottom-left, or bottom-right corners.
    """
    if frame_w <= 0 or frame_h <= 0:
        return None

    xc = x + w / 2.0
    yc = y + h / 2.0

    # Central Gameplay Dead Zone: central 40% of width and central 40% of height
    if (0.30 * frame_w <= xc <= 0.70 * frame_w) or (0.32 * frame_h <= yc <= 0.68 * frame_h):
        return None

    # Facecams are strictly in the four extreme outer corners
    # Corner width: within outer 30% of frame width
    # Corner height: within outer 32% of frame height (rejects middle-edge HUD/character icons)
    is_left = xc < 0.30 * frame_w
    is_right = xc > 0.70 * frame_w
    is_top = yc < 0.32 * frame_h
    is_bottom = yc > 0.68 * frame_h

    if is_left and is_top:
        return "top_left"
    if is_right and is_top:
        return "top_right"
    if is_left and is_bottom:
        return "bottom_left"
    if is_right and is_bottom:
        return "bottom_right"

    # Any face not in the 4 extreme corners is rejected
    return None


class FacecamDetector:
    """Detects streamer facecam regions in 16:9 gameplay streams with anti-false-positive filtering."""

    def __init__(self) -> None:
        self.cascade = None
        try:
            if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as exc:
            logger.warning(f"[Facecam] Could not load face cascade classifier: {exc}")

    async def detect(self, video_path: Path, sample_count: int = 15) -> Optional[FacecamBox]:
        """Async wrapper to detect facecam box across sampled frames."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._detect_sync, video_path, sample_count
        )

    def _detect_sync(self, video_path: Path, sample_count: int = 15) -> Optional[FacecamBox]:
        if self.cascade is None or getattr(self.cascade, "empty", lambda: True)():
            return None
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

        step = max(20, total_frames // (sample_count + 1)) if total_frames > 0 else 45
        valid_corner_detections: list[tuple[int, int, int, int, str]] = []

        for i in range(1, sample_count + 1):
            frame_idx = i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(int(height * 0.08), int(height * 0.08)),
                maxSize=(int(height * 0.45), int(height * 0.45)),
            )

            for (x, y, w, h) in faces:
                quadrant = classify_face_quadrant(x, y, w, h, width, height)
                if quadrant is not None:
                    valid_corner_detections.append((x, y, w, h, quadrant))
                else:
                    logger.debug(f"[Facecam] Rejected face candidate at ({x},{y},{w}x{h}): inside central dead-zone")

        cap.release()

        if not valid_corner_detections:
            logger.info(f"[Facecam] No persistent corner facecam found in {video_path.name}")
            return None

        # Group detections by quadrant to identify dominant corner
        quadrant_counts: dict[str, list[tuple[int, int, int, int]]] = {}
        for (x, y, w, h, q) in valid_corner_detections:
            quadrant_counts.setdefault(q, []).append((x, y, w, h))

        # Find dominant quadrant
        dominant_quadrant, dominant_boxes = max(quadrant_counts.items(), key=lambda item: len(item[1]))

        # Temporal stability check: must be seen in at least 40% of sampled frames
        min_required_frames = max(3, int(sample_count * 0.40))
        if len(dominant_boxes) < min_required_frames:
            logger.info(
                f"[Facecam] Corner detections in '{dominant_quadrant}' ({len(dominant_boxes)}) "
                f"below stability threshold ({min_required_frames})"
            )
            return None

        # Spatial consistency check: standard deviation of position
        xs = [b[0] for b in dominant_boxes]
        ys = [b[1] for b in dominant_boxes]
        ws = [b[2] for b in dominant_boxes]
        hs = [b[3] for b in dominant_boxes]

        std_x = float(np.std(xs))
        std_y = float(np.std(ys))

        # If the detected face moves erratically (>35px std dev), it is in-game motion, not a webcam
        if std_x > 35.0 or std_y > 35.0:
            logger.info(f"[Facecam] Rejected corner face in '{dominant_quadrant}': erratic motion (std_x={std_x:.1f}, std_y={std_y:.1f})")
            return None

        median_x = int(np.median(xs))
        median_y = int(np.median(ys))
        median_w = int(np.median(ws))
        median_h = int(np.median(hs))

        # Aspect ratio check: webcam should be roughly square or horizontal
        aspect_ratio = median_w / max(1, median_h)
        if aspect_ratio < 0.7 or aspect_ratio > 2.2:
            logger.info(f"[Facecam] Rejected box with abnormal aspect ratio: {aspect_ratio:.2f}")
            return None

        # Frame the full webcam container (head + chin + neck + shoulders + background frame)
        # Prevents cutting off streamer mouth, chin, and chest by gameplay crop
        face_size = max(median_w, median_h)
        face_center_x = median_x + median_w / 2.0
        face_center_y = median_y + median_h / 2.0

        # Standard streamer webcam box in 16:9 aspect ratio (~28% of stream width)
        target_cam_w = max(int(face_size * 4.5), int(width * 0.28))
        target_cam_h = int(target_cam_w * 9 / 16)
        target_cam_w = (target_cam_w // 2) * 2
        target_cam_h = (target_cam_h // 2) * 2

        # Position face in upper half of webcam box (face center at ~38% from top of webcam)
        cam_x = int(face_center_x - target_cam_w / 2.0)
        cam_y = int(face_center_y - 0.38 * target_cam_h)

        # Snap to outer stream borders if close to margins
        margin_x = int(width * 0.06)
        margin_y = int(height * 0.06)

        if dominant_quadrant in ("top_left", "bottom_left") and cam_x < margin_x:
            cam_x = 0
        elif dominant_quadrant in ("top_right", "bottom_right") and (width - (cam_x + target_cam_w)) < margin_x:
            cam_x = width - target_cam_w

        if dominant_quadrant in ("top_left", "top_right") and cam_y < margin_y:
            cam_y = 0
        elif dominant_quadrant in ("bottom_left", "bottom_right") and (height - (cam_y + target_cam_h)) < margin_y:
            cam_y = height - target_cam_h

        # Strictly clamp within video dimensions
        cam_x = max(0, min(cam_x, width - target_cam_w))
        cam_y = max(0, min(cam_y, height - target_cam_h))
        cam_w = min(target_cam_w, width - cam_x)
        cam_h = min(target_cam_h, height - cam_y)
        cam_w = (cam_w // 2) * 2
        cam_h = (cam_h // 2) * 2

        # Calculate composite confidence score (0.0 - 1.0)
        detection_ratio = len(dominant_boxes) / sample_count
        jitter_penalty = min(1.0, (std_x + std_y) / 50.0)
        confidence = float(np.clip(detection_ratio * (1.0 - 0.4 * jitter_penalty), 0.0, 1.0))

        logger.info(
            f"[Facecam] Locked corner facecam in '{dominant_quadrant}': "
            f"{cam_w}x{cam_h} at ({cam_x},{cam_y}), confidence={confidence:.2f}"
        )

        return FacecamBox(
            x=cam_x,
            y=cam_y,
            w=cam_w,
            h=cam_h,
            confidence=confidence,
            quadrant=dominant_quadrant,
        )
