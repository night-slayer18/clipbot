"""Multilingual word-level animated karaoke subtitle generator using Whisper and ASS.

Supports English, Hindi, Malayalam, Spanish, and any language recognized by Whisper.
Also handles dialogue-free/slapstick audio (e.g. Tom & Jerry cartoons) gracefully.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class SpokenWord:
    word: str
    start: float  # seconds
    end: float    # seconds


def _format_ass_time(seconds: float) -> str:
    """Format seconds into ASS timestamp format: H:MM:SS.cs"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hrs}:{mins:02d}:{secs:02d}.{centis:02d}"


# Multilingual ASS header with universal unicode font fallback (Noto Sans / Arial Unicode)
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans,Arial,66,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,3,2,40,40,420,1
Style: Highlight,Noto Sans,Arial,72,&H0000FFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,105,105,0,0,1,6,4,2,40,40,420,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


class CaptionGenerator:
    """Transcribes video audio and produces styled karaoke ASS subtitles in any language."""

    def __init__(self, model_size: str = "base") -> None:
        # Default to multilingual 'base' (supports English, Hindi, Malayalam, 90+ languages)
        self.model_size = model_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                logger.info(f"[Captions] Loading multilingual Whisper model '{self.model_size}'")
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            except ImportError:
                logger.warning("[Captions] faster-whisper not installed; subtitles unavailable")
                return None
        return self._model

    async def generate_ass(
        self,
        video_path: Path,
        output_ass: Path,
        max_words_per_line: int = 4,
        language: Optional[str] = None,
    ) -> Optional[Path]:
        """Generate an .ass subtitle file with word-by-word karaoke pop."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._generate_ass_sync, video_path, output_ass, max_words_per_line, language
        )

    def _generate_ass_sync(
        self,
        video_path: Path,
        output_ass: Path,
        max_words_per_line: int = 4,
        language: Optional[str] = None,
    ) -> Optional[Path]:
        model = self._get_model()
        if model is None:
            return None

        try:
            # Auto-detect language if None (works for English, Hindi, Malayalam, etc.)
            segments, info = model.transcribe(
                str(video_path),
                beam_size=3,
                word_timestamps=True,
                vad_filter=True,
                language=language,
            )

            detected_lang = getattr(info, "language", "unknown")
            logger.info(f"[Captions] Detected language for {video_path.name}: {detected_lang}")

            all_words: list[SpokenWord] = []
            for seg in segments:
                if seg.words:
                    for w in seg.words:
                        cleaned = w.word.strip()
                        if cleaned:
                            # Keep original casing/unicode for Hindi/Malayalam; uppercase for Latin
                            word_display = cleaned if any(ord(c) > 127 for c in cleaned) else cleaned.upper()
                            all_words.append(SpokenWord(word=word_display, start=w.start, end=w.end))

            if not all_words:
                logger.info(f"[Captions] No speech detected in {video_path.name} (cartoon/music content)")
                return None

            # Group words into short punchy chunks (2-4 words per line for mobile vertical short format)
            chunks: list[list[SpokenWord]] = []
            current_chunk: list[SpokenWord] = []

            for w in all_words:
                current_chunk.append(w)
                if len(current_chunk) >= max_words_per_line:
                    chunks.append(current_chunk)
                    current_chunk = []
                elif len(current_chunk) > 1 and (w.start - current_chunk[-2].end > 0.6):
                    chunks.append(current_chunk[:-1])
                    current_chunk = [w]

            if current_chunk:
                chunks.append(current_chunk)

            # Write ASS events
            output_ass.parent.mkdir(parents=True, exist_ok=True)
            lines = [ASS_HEADER]

            for chunk in chunks:
                if not chunk:
                    continue
                for i, active_w in enumerate(chunk):
                    start_str = _format_ass_time(active_w.start)
                    end_str = _format_ass_time(active_w.end)

                    parts = []
                    for j, w in enumerate(chunk):
                        # Sanitize ASS special formatting characters from word
                        clean_word = w.word.replace("{", "").replace("}", "").replace("\\", "")
                        if i == j:
                            # Highlighted active word: bright yellow (&H00E6FF& in BGR)
                            parts.append(r"{\c&H00E6FF&\fscx108\fscy108}" + clean_word + r"{\r}")
                        else:
                            parts.append(clean_word)

                    line_text = " ".join(parts)
                    lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{line_text}\n")

            with output_ass.open("w", encoding="utf-8") as f:
                f.writelines(lines)

            logger.info(f"[Captions] Generated animated subtitles ({len(all_words)} words, lang: {detected_lang})")
            return output_ass

        except Exception as exc:
            logger.warning(f"[Captions] Subtitle generation failed: {exc}")
            return None
