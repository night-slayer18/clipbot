"""Audio detector — extracts vocal-gated RMS loudness, onset bursts, Silero VAD, and vocal pitch jumps."""

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
        cache_file = media_path.with_name(f"{media_path.stem}_audio_signals.npz")
        if cache_file.exists():
            try:
                data = np.load(str(cache_file))
                logger.info(f"[Audio] Loaded cached signals from {cache_file.name}")
                return {k: data[k] for k in data.files}
            except Exception as e:
                logger.warning(f"[Audio] Failed loading cache {cache_file}: {e}")

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
                    "audio_vad": np.zeros(dur, dtype=np.float32),
                    "audio_pitch": np.zeros(dur, dtype=np.float32),
                }

            # Run blocking librosa audio analysis in thread pool
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(None, self._analyze_audio_sync, wav_path)

            logger.debug(f"[Audio] {asset.external_id}: {len(res['audio_rms'])}s analyzed")
            try:
                np.savez_compressed(str(cache_file), **res)
            except Exception as e:
                logger.debug(f"[Audio] Failed saving cache {cache_file}: {e}")
            return res

        finally:
            wav_path.unlink(missing_ok=True)

    def _analyze_audio_sync(self, wav_path: Path) -> dict[str, np.ndarray]:
        import librosa

        y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
        frame_len = sr
        n_frames = max(1, len(y) // frame_len)

        # 1. Base RMS energy in 1-second frames
        rms = np.array([
            np.sqrt(np.mean(y[i * frame_len:(i + 1) * frame_len] ** 2))
            for i in range(n_frames)
        ], dtype=np.float32)

        # 2. Silero VAD Voice Activity Gating
        vad_mask = np.zeros(n_frames, dtype=np.float32)
        try:
            from faster_whisper.vad import get_speech_timestamps, VadOptions
            speech_ts = get_speech_timestamps(
                y,
                vad_options=VadOptions(threshold=0.5, min_speech_duration_ms=250),
                sampling_rate=16000,
            )
            for seg in speech_ts:
                s_sec = int(seg["start"] / 16000)
                e_sec = int(np.ceil(seg["end"] / 16000))
                for sec in range(max(0, s_sec), min(n_frames, e_sec)):
                    vad_mask[sec] = 1.0
        except Exception as e:
            logger.debug(f"[Audio] Silero VAD fallback: {e}")
            vad_mask = np.ones(n_frames, dtype=np.float32)

        # Vocal-weighted RMS: speech gets full 1.0x weight; pure silence/ambient game audio gets 0.2x weight
        vocal_rms = rms * (0.2 + 0.8 * vad_mask)

        # 3. Onset burst (spectral flux) in 1-second frames
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        if len(onset) > 0:
            onset_1hz = np.interp(
                np.linspace(0, len(onset), n_frames),
                np.arange(len(onset)),
                onset,
            ).astype(np.float32)
        else:
            onset_1hz = np.zeros(n_frames, dtype=np.float32)

        # 4. F0 Fundamental Vocal Pitch Jumps (Screams / Excitement Spikes)
        pitch_spike = np.zeros(n_frames, dtype=np.float32)
        try:
            # Quick fundamental pitch via YIN
            f0 = librosa.yin(y, fmin=65, fmax=550, sr=sr, frame_length=2048, hop_length=sr)
            if len(f0) > 0:
                f0_1hz = f0[:n_frames]
                valid_f0 = f0_1hz[~np.isnan(f0_1hz)]
                if len(valid_f0) > 3:
                    med_pitch = np.median(valid_f0)
                    std_pitch = np.std(valid_f0) + 1e-5
                    pitch_z = (f0_1hz - med_pitch) / std_pitch
                    pitch_spike = np.clip(np.nan_to_num(pitch_z, nan=0.0) * vad_mask, 0.0, 5.0).astype(np.float32)
        except Exception as e:
            logger.debug(f"[Audio] Pitch estimation fallback: {e}")

        return {
            "audio_rms": vocal_rms,
            "audio_onset": onset_1hz,
            "audio_vad": vad_mask,
            "audio_pitch": pitch_spike,
        }
