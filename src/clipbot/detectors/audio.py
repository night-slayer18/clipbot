"""Audio detector — extracts RMS loudness and onset burst signals."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger

from clipbot.core.models import Asset


class AudioDetector:
    name = "audio"

    async def signals(self, asset: Asset, media_path: Path) -> dict[str, np.ndarray]:
        # Extract audio as mono 16kHz wav via ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(media_path),
                "-vn", "-ac", "1", "-ar", "16000",
                "-f", "wav", str(wav_path)
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
            )
            await proc.wait()

            if proc.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size < 100:
                logger.debug(f"[Audio] No valid audio track extracted for {asset.external_id}")
                dur = int(asset.duration or 60)
                return {
                    "audio_rms": np.zeros(dur, dtype=np.float32),
                    "audio_onset": np.zeros(dur, dtype=np.float32),
                }

            # Run blocking librosa audio analysis in thread pool
            loop = asyncio.get_event_loop()
            rms, onset_1hz = await loop.run_in_executor(None, self._analyze_audio_sync, wav_path)

            logger.debug(f"[Audio] {asset.external_id}: {len(rms)}s analyzed")
            return {"audio_rms": rms, "audio_onset": onset_1hz}

        finally:
            wav_path.unlink(missing_ok=True)

    def _analyze_audio_sync(self, wav_path: Path) -> tuple[np.ndarray, np.ndarray]:
        import librosa

        y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
        frame_len = sr
        n_frames = max(1, len(y) // frame_len)

        # RMS energy in 1-second frames
        rms = np.array([
            np.sqrt(np.mean(y[i * frame_len:(i + 1) * frame_len] ** 2))
            for i in range(n_frames)
        ], dtype=np.float32)

        # Onset burst (spectral flux) in 1-second frames
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        if len(onset) > 0:
            onset_1hz = np.interp(
                np.linspace(0, len(onset), n_frames),
                np.arange(len(onset)),
                onset,
            ).astype(np.float32)
        else:
            onset_1hz = np.zeros(n_frames, dtype=np.float32)

        return rms, onset_1hz
