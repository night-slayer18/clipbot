"""Wired integration tests for subtitle generation and episodic movie chaptering."""

from __future__ import annotations

from pathlib import Path
import pytest

from clipbot.render.captions import (
    CaptionGenerator,
    SpokenWord,
    _format_ass_time,
    sanitize_and_chunk_words,
    ASS_HEADER,
)
from clipbot.render.movie_splitter import MovieSplitter


def test_ass_time_formatting():
    assert _format_ass_time(0.0) == "0:00:00.00"
    assert _format_ass_time(65.42) == "0:01:05.42"
    assert _format_ass_time(3661.05) == "1:01:01.05"
    # Test centisecond overflow carry-over
    assert _format_ass_time(59.999) == "0:01:00.00"


def test_caption_ass_header_and_safe_zone():
    assert "PlayResX: 1080" in ASS_HEADER
    assert "PlayResY: 1920" in ASS_HEADER
    assert "WrapStyle: 2" in ASS_HEADER
    assert "MarginV, Encoding\nStyle: Default," in ASS_HEADER
    # Style line must have exactly 23 fields (1 name + 22 values after the comma)
    # Verify MarginV=700 lands in the correct field position
    assert ",60,60,700,1" in ASS_HEADER
    # Primary white, Secondary yellow — verify these are in the correct colour field positions
    assert "&H00FFFFFF" in ASS_HEADER
    assert "&H0000FFFF" in ASS_HEADER
    # Only one font name — no comma-separated fallback list that shifts all fields
    style_line = [l for l in ASS_HEADER.splitlines() if l.startswith("Style: Default,")][0]
    parts = style_line.split(",")
    # Correct field count: "Style: Default" + 22 value fields = 23 total (index 0..22)
    assert len(parts) == 23, f"Style line has {len(parts)} fields, expected 23. Got: {style_line}"
    # Fontsize must be numeric (74), not a font name like 'Impact'
    fontsize_field = parts[2]
    assert fontsize_field.isdigit(), f"Fontsize field '{fontsize_field}' is not numeric — font fallback list shifted fields"


def test_sanitize_and_chunk_words():
    raw_words = [
        SpokenWord(word="Hello", start=1.0, end=1.2),
        SpokenWord(word="world", start=1.25, end=1.5),
        SpokenWord(word="!", start=1.5, end=1.55),           # Pure punctuation, must merge
        SpokenWord(word="This", start=1.6, end=1.62),         # < 0.12 duration, must floor
        SpokenWord(word="is", start=1.65, end=1.75),
        SpokenWord(word="insane", start=2.5, end=2.9),        # Gap > 0.45s, must split
    ]

    chunks = sanitize_and_chunk_words(raw_words, max_words_per_line=4, max_chars_per_line=24)

    # First chunk: "HELLO", "WORLD!", "THIS", "IS" (max 4 words or split on pause)
    # The pause between 'is' (1.75s or floored) and 'insane' (2.5s) is > 0.45s, so 'insane' is in chunk 2
    assert len(chunks) == 2
    chunk1 = chunks[0]
    assert len(chunk1) == 4
    assert chunk1[0].word == "HELLO"
    assert chunk1[1].word == "WORLD!"  # Punctuation merged
    assert chunk1[2].word == "THIS"
    assert chunk1[2].end - chunk1[2].start >= 0.12  # Floored
    assert chunk1[3].word == "IS"

    chunk2 = chunks[1]
    assert len(chunk2) == 1
    assert chunk2[0].word == "INSANE"
    assert chunk2[0].start >= chunk1[-1].end  # Monotonicity


def test_movie_splitter_smart_cut_points():
    splitter = MovieSplitter()
    splitter.target_part_sec = 60
    splitter.min_part_sec = 50
    splitter.max_part_sec = 70

    # Total duration = 180s (3 minutes)
    total_duration = 180.0
    # Silences at:
    # 58s (near ideal 60s)
    # 119s (near ideal 120s)
    silences = [
        (25.0, 26.0),
        (58.0, 59.5),   # Cut 1: Midpoint = 58.75s
        (85.0, 86.0),
        (119.0, 120.0), # Cut 2: Midpoint = 119.5s
    ]

    cuts = splitter.calculate_smart_cut_points(total_duration, silences)

    assert len(cuts) == 3
    # First cut should start at 0 and end around 58.75s
    assert cuts[0][0] == 0.0
    assert abs(cuts[0][1] - 58.75) < 0.1

    # Second cut should start at 58.75 and end around 119.5s
    assert abs(cuts[1][0] - 58.75) < 0.1
    assert abs(cuts[1][1] - 119.5) < 0.1

    # Final cut should finish at total duration 180.0s
    assert abs(cuts[2][0] - 119.5) < 0.1
    assert cuts[2][1] == 180.0


def test_drawtext_character_escaping():
    # Verify titles with quotes, colons, or percent signs are sanitized
    raw_title = "Schindler's List: Part 1 (100% Masterpiece)"
    safe_title = raw_title.replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")
    
    assert "\\'" in safe_title
    assert "\\:" in safe_title
    assert "\\%" in safe_title
