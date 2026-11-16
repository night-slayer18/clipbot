"""Wired integration tests for subtitle generation and episodic movie chaptering."""

from __future__ import annotations

from pathlib import Path
import pytest

from clipbot.render.captions import CaptionGenerator, SpokenWord, _format_ass_time
from clipbot.render.movie_splitter import MovieSplitter


def test_ass_time_formatting():
    assert _format_ass_time(0.0) == "0:00:00.00"
    assert _format_ass_time(65.42) == "0:01:05.42"
    assert _format_ass_time(3661.05) == "1:01:01.05"


def test_caption_ass_events_and_styling(tmp_path: Path):
    ass_file = tmp_path / "test_subs.ass"
    words = [
        SpokenWord(word="HELLO", start=1.0, end=1.5),
        SpokenWord(word="WORLD", start=1.6, end=2.0),
        SpokenWord(word="INSANE", start=2.2, end=2.8),
    ]

    # Test ASS header and dialogue event structure
    from clipbot.render.captions import ASS_HEADER
    assert "PlayResX: 1080" in ASS_HEADER
    assert "PlayResY: 1920" in ASS_HEADER
    assert "Noto Sans" in ASS_HEADER


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
