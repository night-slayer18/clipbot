"""Multilingual word-level animated karaoke subtitle generator using Whisper and ASS.

Supports English, Hindi, Malayalam, Spanish, and any language recognized by Whisper.
Also handles dialogue-free/slapstick audio (e.g. Tom & Jerry cartoons) gracefully.
Hardened with anti-hallucination shields and mobile safe-zone positioning.
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
    """Format seconds into ASS timestamp format: H:MM:SS.cs with proper carry-over."""
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis = 0
        secs += 1
        if secs >= 60:
            secs = 0
            mins += 1
            if mins >= 60:
                mins = 0
                hrs += 1
    return f"{hrs}:{mins:02d}:{secs:02d}.{centis:02d}"


def sanitize_and_chunk_words(
    raw_words: list[SpokenWord],
    max_words_per_line: int = 4,
    max_chars_per_line: int = 24,
    min_word_duration: float = 0.12,  # 12 centiseconds floor
) -> list[list[SpokenWord]]:
    """Sanitize raw Whisper words and group into short, stable lines for mobile 9:16 screens.

    1. Merge standalone punctuation into the previous word.
    2. Enforce monotonic progression: start_{i+1} >= end_i.
    3. Floor minimum duration to avoid 0-duration ASS rendering bugs.
    4. Group into bounded chunks (max 3-4 words, max 24 chars) to prevent line wrapping jitter.
    5. Split chunks on natural speech pauses (> 0.45s).
    """
    if not raw_words:
        return []

    # 1. Merge punctuation tokens
    merged: list[SpokenWord] = []
    for w in raw_words:
        clean = w.word.strip()
        if not clean:
            continue
        # Check if pure punctuation / symbols
        if not any(c.isalnum() for c in clean) and merged:
            merged[-1].word += clean
            merged[-1].end = max(merged[-1].end, w.end)
        else:
            # Uppercase Latin words for viral shorts punchiness; preserve Unicode scripts (Hindi, Malayalam, etc.)
            display_word = clean if any(ord(c) > 127 for c in clean) else clean.upper()
            # Strip problematic ASS markup braces from word text
            display_word = display_word.replace("{", "").replace("}", "").replace("\\", "")
            if display_word:
                merged.append(SpokenWord(word=display_word, start=w.start, end=w.end))

    if not merged:
        return []

    # 2. Timing monotonicity & duration floor
    sanitized: list[SpokenWord] = []
    prev_end = 0.0
    for w in merged:
        w_start = max(prev_end, w.start)
        w_end = max(w_start + min_word_duration, w.end)
        sanitized.append(SpokenWord(word=w.word, start=w_start, end=w_end))
        prev_end = w_end

    # 3. Chunking for mobile display
    chunks: list[list[SpokenWord]] = []
    current_chunk: list[SpokenWord] = []
    current_chars = 0

    for w in sanitized:
        pause = (w.start - current_chunk[-1].end) if current_chunk else 0.0
        should_split = (
            len(current_chunk) >= max_words_per_line
            or (current_chars + len(w.word) > max_chars_per_line and current_chunk)
            or pause > 0.45
        )
        if should_split and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0

        current_chunk.append(w)
        current_chars += len(w.word) + 1  # count space

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# Multilingual ASS header with universal unicode font fallback and mobile safe-zone positioning
# MarginV: 700 with Alignment: 2 anchors the text baseline at Y = 1920 - 700 = 1220 (Universal Safe Band)
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,Impact,Noto Sans,Arial,74,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,2,60,60,700,1

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
        max_chars_per_line: int = 24,
        language: Optional[str] = None,
    ) -> Optional[Path]:
        """Generate an .ass subtitle file with word-by-word karaoke pop."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._generate_ass_sync, video_path, output_ass, max_words_per_line, max_chars_per_line, language
        )

    def _generate_ass_sync(
        self,
        video_path: Path,
        output_ass: Path,
        max_words_per_line: int = 4,
        max_chars_per_line: int = 24,
        language: Optional[str] = None,
    ) -> Optional[Path]:
        model = self._get_model()
        if model is None:
            return None

        try:
            vad_params = {
                "threshold": 0.55,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 400,
            }
            initial_prompt = "GTA RP, NoPixel, clutch, pog, one-shot, griefing, wasted, diffed, ace, caster, clip, pwoli, bhai, ekdum hard, vera level, vamos"

            # Auto-detect language if None (works for English, Hindi, Malayalam, etc.)
            segments, info = model.transcribe(
                str(video_path),
                beam_size=3,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=vad_params,
                condition_on_previous_text=False,
                temperature=0.0,
                no_speech_threshold=0.60,
                compression_ratio_threshold=2.4,
                initial_prompt=initial_prompt,
                language=language,
            )

            detected_lang = getattr(info, "language", "unknown")
            logger.info(f"[Captions] Detected language for {video_path.name}: {detected_lang}")

            raw_words: list[SpokenWord] = []
            for seg in segments:
                if seg.words:
                    for w in seg.words:
                        if w.word.strip():
                            raw_words.append(SpokenWord(word=w.word, start=w.start, end=w.end))

            if not raw_words:
                logger.info(f"[Captions] No speech detected in {video_path.name} (cartoon/music content)")
                return None

            chunks = sanitize_and_chunk_words(
                raw_words,
                max_words_per_line=max_words_per_line,
                max_chars_per_line=max_chars_per_line,
            )

            if not chunks:
                return None

            # Write ASS events
            output_ass.parent.mkdir(parents=True, exist_ok=True)
            lines = [ASS_HEADER]

            for chunk in chunks:
                if not chunk:
                    continue
                for i, active_w in enumerate(chunk):
                    # To prevent zero-frame subtitle flicker between words in a chunk,
                    # make each word active until the start of the next word (or end of chunk)
                    start_str = _format_ass_time(active_w.start)
                    if i + 1 < len(chunk):
                        end_t = max(active_w.end, chunk[i + 1].start)
                    else:
                        end_t = active_w.end
                    end_str = _format_ass_time(end_t)

                    parts = []
                    for j, w in enumerate(chunk):
                        if i == j:
                            # Highlighted active word: bright yellow (&H0000FFFF&) with 12% scale pop
                            parts.append(r"{\c&H0000FFFF&\fscx112\fscy112}" + w.word + r"{\r}")
                        else:
                            parts.append(w.word)

                    line_text = " ".join(parts)
                    lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{line_text}\n")

            with output_ass.open("w", encoding="utf-8") as f:
                f.writelines(lines)

            total_words = sum(len(c) for c in chunks)
            logger.info(f"[Captions] Generated animated subtitles ({total_words} words, lang: {detected_lang})")
            return output_ass

        except Exception as exc:
            logger.warning(f"[Captions] Subtitle generation failed: {exc}")
            return None
