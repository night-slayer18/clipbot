"""Transcript detector — runs faster-whisper on candidate windows, scores keywords.

Multilingual support: English, Hindi, Malayalam, and more.
Gracefully handles cartoons / dialogue-free content (like Tom & Jerry).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from loguru import logger

from clipbot.core.config import load_profile
from clipbot.core.models import Asset


# Multilingual hype keywords
DEFAULT_HIGH_KEYWORDS = [
    # English
    "wasted", "busted", "no way", "oh my god", "run", "shoot", "he's dead", "knockout", "goal", "dunk", "six", "wicket",
    # Hindi
    "bhai", "kya baat", "shandar", "mar gaya", "khatam", "arre yaar", "dekho", "chhakka", "wicket",
    # Malayalam
    "ente daivame", "adipoli", "theernnu", "marichu", "poyi", "machane", "nokku", "adipoli goal",
]
DEFAULT_MED_KEYWORDS = [
    # English
    "get in", "drive", "police", "officer", "hands up", "got him", "oh no", "what the", "bro",
    # Hindi
    "chalo", "pakdo", "bhaago", "roko", "bachao", "shabash",
    # Malayalam
    "vaa", "odikkko", "nirthu", "kolaam", "poli",
]

HIGH_WEIGHT = 1.5
MED_WEIGHT = 1.0


def _keyword_score(text: str, high_kw: list[str], med_kw: list[str]) -> float:
    t = text.lower()
    score = 0.0
    for kw in high_kw:
        if kw in t:
            score += HIGH_WEIGHT
    for kw in med_kw:
        if kw in t:
            score += MED_WEIGHT
    return score


class TranscriptDetector:
    name = "transcript"

    def __init__(self, model_size: str = "base", device: str = "auto") -> None:
        self._model_size = model_size
        self._device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
            device = "cpu"
            if self._device != "auto":
                device = self._device
            logger.info(f"[Transcript] Loading multilingual Whisper model '{self._model_size}' on {device}")
            self._model = WhisperModel(self._model_size, device=device, compute_type="int8")
        except ImportError:
            logger.warning("[Transcript] faster-whisper not installed — transcript detector disabled")
            self._model = None
        return self._model

    async def signals(self, asset: Asset, media_path: Path) -> dict[str, np.ndarray]:
        result = await asyncio.get_event_loop().run_in_executor(
            None, self._transcribe_sync, media_path, asset
        )
        return result

    def _transcribe_sync(self, media_path: Path, asset: Asset) -> dict[str, np.ndarray]:
        model = self._load_model()
        if model is None:
            dur = int(asset.duration or 60)
            return {"transcript_hype": np.zeros(dur, dtype=np.float32)}

        high_kw = DEFAULT_HIGH_KEYWORDS
        med_kw = DEFAULT_MED_KEYWORDS

        try:
            logger.debug(f"[Transcript] Transcribing {media_path.name}")
            segments, info = model.transcribe(
                str(media_path),
                beam_size=3,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )

            # Determine actual duration from Whisper audio info if available
            audio_dur = getattr(info, "duration", None)
            duration = int(max(audio_dur or 0, asset.duration or 0, 60))
            hype = np.zeros(duration, dtype=np.float32)

            for seg in segments:
                mid_sec = int((seg.start + seg.end) / 2)
                score = _keyword_score(seg.text, high_kw, med_kw)
                if score > 0 and 0 <= mid_sec < duration:
                    start_i = max(0, int(seg.start))
                    end_i = min(duration, int(seg.end) + 1)
                    hype[start_i:end_i] += score / max(1, end_i - start_i)

            return {"transcript_hype": hype}

        except Exception as exc:
            logger.warning(f"[Transcript] Transcription notice for {asset.external_id}: {exc}")
            dur = int(asset.duration or 60)
            return {"transcript_hype": np.zeros(dur, dtype=np.float32)}
