"""Transcript detector — runs faster-whisper on candidate windows, scores keywords.

Multilingual support: English, Hindi, Malayalam, Spanish, and more.
Gracefully handles cartoons / dialogue-free content (like Tom & Jerry).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
from loguru import logger

from clipbot.core.config import load_profile
from clipbot.core.media import probe_media_duration
from clipbot.core.models import Asset

# Multilingual hype keywords across English, Hindi/Hinglish, Malayalam/Manglish, and Spanish
DEFAULT_HIGH_KEYWORDS = [
    # English Gaming & Hype
    "wasted", "busted", "no way", "oh my god", "run", "shoot", "he's dead", "dead", "kill", "knockout", "goal", "dunk", "six", "wicket",
    "clutch", "ace", "one shot", "one-shot", "diffed", "lets go", "let's go", "headshot", "insane", "holy shit", "unbelievable",
    # Hindi & Hinglish
    "bhai", "kya baat", "shandar", "mar gaya", "khatam", "arre yaar", "dekho", "chhakka", "wicket",
    "ekdum hard", "phod diya", "faad diya", "lag gaye", "hag diya", "noobda", "op play", "rush kar", "one shot hai", "kya kar raha hai",
    # Malayalam & Manglish
    "ente daivame", "adipoli", "theernnu", "marichu", "poyi", "machane", "nokku", "adipoli goal",
    "pwoli", "pwolichu", "mass play", "vera level", "tholppikkan patilla", "kalakki", "katte play", "manda", "chathi",
    # Spanish
    "dios mio", "no me lo creo", "que locura", "vamos", "partidaza", "eliminado", "muerto", "brutal", "toma ya", "no puede ser",
]
DEFAULT_MED_KEYWORDS = [
    # English
    "get in", "drive", "police", "officer", "hands up", "got him", "oh no", "what the", "bro", "pog", "poggers", "clip it",
    # Hindi
    "chalo", "pakdo", "bhaago", "roko", "bachao", "shabash", "jaldi", "dhyan se",
    # Malayalam
    "vaa", "odikkko", "nirthu", "kolaam", "poli", "vegam",
    # Spanish
    "cuidado", "rapido", "mira esto", "buena", "dale",
]

HIGH_WEIGHT = 1.5
MED_WEIGHT = 1.0


def _keyword_score(text: str, high_kw: list[str], med_kw: list[str]) -> float:
    """Score text by matching whole keywords using word boundaries, avoiding substring collisions."""
    t = text.lower()
    score = 0.0
    for kw in high_kw:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, t):
            score += HIGH_WEIGHT
    for kw in med_kw:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, t):
            score += MED_WEIGHT
    return score


def _get_active_keywords(profile_name: str | None = None) -> tuple[list[str], list[str]]:
    """Merge profile-specific keywords with universal defaults."""
    high = list(DEFAULT_HIGH_KEYWORDS)
    med = list(DEFAULT_MED_KEYWORDS)
    if profile_name:
        try:
            profile = load_profile(profile_name)
            prof_kw = profile.get("transcript_keywords", {})
            for k in prof_kw.get("high", []):
                high.append(k.lower())
            for k in prof_kw.get("medium", []):
                med.append(k.lower())
        except Exception:
            pass
    return list(set(high)), list(set(med))


class TranscriptDetector:
    name = "transcript"

    def __init__(self, model_size: str = "base", device: str = "auto", profile_name: str = "general_viral") -> None:
        self._model_size = model_size
        self._device = device
        self._profile_name = profile_name
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

    async def signals(
        self,
        asset: Asset,
        media_path: Path,
        profile_name: str | None = None,
        candidate_windows: list[tuple[float, float]] | None = None,
    ) -> dict[str, np.ndarray]:
        active_profile = profile_name or self._profile_name
        result = await asyncio.get_event_loop().run_in_executor(
            None, self._transcribe_sync, media_path, asset, active_profile, candidate_windows
        )
        return result

    def _transcribe_sync(
        self,
        media_path: Path,
        asset: Asset,
        profile_name: str,
        candidate_windows: list[tuple[float, float]] | None = None,
    ) -> dict[str, np.ndarray]:
        probed_dur = probe_media_duration(media_path)
        actual_duration = int(probed_dur or asset.duration or 60)

        model = self._load_model()
        if model is None:
            return {"transcript_hype": np.zeros(actual_duration, dtype=np.float32)}

        high_kw, med_kw = _get_active_keywords(profile_name)

        cache_file = media_path.with_name(f"{media_path.stem}_transcript_signals.npz")
        if cache_file.exists():
            try:
                data = np.load(str(cache_file))
                logger.info(f"[Transcript] Loaded cached signals from {cache_file.name}")
                return {k: data[k] for k in data.files}
            except Exception as e:
                logger.warning(f"[Transcript] Failed loading cache {cache_file}: {e}")

        try:
            logger.debug(f"[Transcript] Transcribing {media_path.name} with profile '{profile_name}' (windows={len(candidate_windows) if candidate_windows else 'all'})")
            initial_prompt = "GTA RP, NoPixel, clutch, pog, one-shot, griefing, wasted, diffed, ace, caster, clip, pwoli, bhai, ekdum hard, vera level, vamos"
            transcribe_kwargs = {
                "beam_size": 1,
                "word_timestamps": True,
                "condition_on_previous_text": False,
                "temperature": 0.0,
                "no_speech_threshold": 0.60,
                "compression_ratio_threshold": 2.4,
                "initial_prompt": initial_prompt,
            }

            duration = actual_duration
            hype = np.zeros(duration, dtype=np.float32)

            if candidate_windows:
                # Transcribe each candidate window audio chunk independently (O(1) memory, avoids OOM on long streams)
                for win_start, win_end in candidate_windows:
                    w_dur = max(1.0, win_end - win_start)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        chunk_path = Path(tmp.name)
                    try:
                        cmd = [
                            "ffmpeg", "-y", "-ss", f"{win_start:.2f}", "-t", f"{w_dur:.2f}",
                            "-i", str(media_path), "-vn", "-ac", "1", "-ar", "16000",
                            "-f", "wav", str(chunk_path)
                        ]
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                        segments, _ = model.transcribe(str(chunk_path), **transcribe_kwargs)
                        for seg in segments:
                            abs_start = win_start + seg.start
                            abs_end = win_start + seg.end
                            mid_sec = int((abs_start + abs_end) / 2)
                            score = _keyword_score(seg.text, high_kw, med_kw)
                            if score > 0 and 0 <= mid_sec < duration:
                                start_i = max(0, int(abs_start))
                                end_i = min(duration, int(abs_end) + 1)
                                hype[start_i:end_i] += score / max(1, end_i - start_i)
                    finally:
                        chunk_path.unlink(missing_ok=True)
            else:
                vad_params = {
                    "threshold": 0.55,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 400,
                }
                transcribe_kwargs["vad_filter"] = True
                transcribe_kwargs["vad_parameters"] = vad_params
                segments, info = model.transcribe(str(media_path), **transcribe_kwargs)
                audio_dur = getattr(info, "duration", None)
                if audio_dur:
                    duration = int(max(audio_dur, duration))
                    if duration > len(hype):
                        hype = np.pad(hype, (0, duration - len(hype)))
                for seg in segments:
                    mid_sec = int((seg.start + seg.end) / 2)
                    score = _keyword_score(seg.text, high_kw, med_kw)
                    if score > 0 and 0 <= mid_sec < duration:
                        start_i = max(0, int(seg.start))
                        end_i = min(duration, int(seg.end) + 1)
                        hype[start_i:end_i] += score / max(1, end_i - start_i)

            res = {"transcript_hype": hype}
            try:
                np.savez_compressed(str(cache_file), **res)
            except Exception as e:
                logger.debug(f"[Transcript] Failed saving cache {cache_file}: {e}")
            return res

        except Exception as exc:
            logger.warning(f"[Transcript] Transcription notice for {asset.external_id}: {exc}")
            return {"transcript_hype": np.zeros(actual_duration, dtype=np.float32)}
