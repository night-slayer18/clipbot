"""Wired integration tests for multi-modal detectors using real generated media."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from clipbot.core.models import Asset
from clipbot.detectors.audio import AudioDetector
from clipbot.detectors.chat import ChatDetector
from clipbot.detectors.facecam import FacecamBox, FacecamDetector, classify_face_quadrant
from clipbot.detectors.transcript import TranscriptDetector, _keyword_score
from clipbot.detectors.visual import VisualDetector, _detect_impact_vignette
from tests.conftest import generate_real_mp4, generate_real_wav


def test_visual_detector_impact_vignette():
    # 1. Normal colored frame (no impact vignette)
    normal_frame = np.full((360, 640, 3), (120, 180, 50), dtype=np.uint8)
    assert _detect_impact_vignette(normal_frame) == 0.0

    # 2. High-impact red vignette / damage flash
    impact_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    impact_frame[:, :, 2] = 160  # Red
    impact_frame[:, :, 1] = 50   # Green
    impact_frame[:, :, 0] = 50   # Blue
    assert _detect_impact_vignette(impact_frame) == 1.0


def test_visual_detector_real_video(tmp_path: Path):
    video_file = tmp_path / "test_visual.mp4"
    generate_real_mp4(video_file, duration_sec=4, fps=15)

    asset = Asset(external_id="test_vis", duration=4.0)
    det = VisualDetector()
    signals = det._analyse_sync(video_file, asset)

    assert "visual_impact" in signals
    assert "visual_scene" in signals
    assert "visual_motion" in signals
    assert "visual_intermission" in signals
    assert len(signals["visual_impact"]) >= 3
    assert len(signals["visual_scene"]) >= 3
    assert len(signals["visual_motion"]) >= 3
    assert len(signals["visual_intermission"]) >= 3


def test_facecam_crop_even_parity():
    # Test that FacecamBox enforces even dimensions for FFmpeg yuv420p
    box = FacecamBox(x=10, y=20, w=345, h=193, confidence=0.85, quadrant="top_right")
    crop_str = box.to_ffmpeg_crop(1920, 1080)
    # w:h:x:y -> 344:192:10:20
    parts = [int(p) for p in crop_str.split(":")]
    assert parts[0] % 2 == 0  # Width is even
    assert parts[1] % 2 == 0  # Height is even
    assert parts[0] == 344
    assert parts[1] == 192


def test_classify_face_quadrant_dead_zone_and_corners():
    # 1. Shroud GTA 6 Bug Face (Center of screen): 309x173 at (611, 450) in 1920x1080
    # Must be REJECTED (returns None)
    assert classify_face_quadrant(611, 450, 309, 173, 1920, 1080) is None

    # 2. Dead center face
    assert classify_face_quadrant(860, 440, 200, 200, 1920, 1080) is None

    # 3. Valid Top-Right corner facecam
    assert classify_face_quadrant(1500, 50, 300, 200, 1920, 1080) == "top_right"

    # 4. Valid Top-Left corner facecam
    assert classify_face_quadrant(50, 50, 300, 200, 1920, 1080) == "top_left"

    # 5. Valid Bottom-Right corner facecam
    assert classify_face_quadrant(1500, 800, 300, 200, 1920, 1080) == "bottom_right"

    # 6. Valid Bottom-Left corner facecam
    assert classify_face_quadrant(50, 800, 300, 200, 1920, 1080) == "bottom_left"


@pytest.mark.asyncio
async def test_chat_detector_velocity_and_emotes(tmp_path: Path):
    chat_file = tmp_path / "chat.jsonl"
    messages = [
        {"offset": 2, "message": "LOL KEKW that was insane 💀"},
        {"offset": 2, "message": "ICANT OMEGALUL"},
        {"offset": 5, "message": "nice shot Pog"},
        {"offset": 10, "message": "normal comment"},
    ]
    with chat_file.open("w") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")

    asset = Asset(external_id="test_chat", duration=15.0, chat_path=chat_file)
    det = ChatDetector()
    signals = await det.signals(asset, Path("/tmp/dummy"))

    assert "chat_velocity" in signals
    velocity = signals["chat_velocity"]
    assert len(velocity) == 15
    # Offset 2 has 2 high-weight emote messages, should have higher velocity than offset 10
    assert velocity[2] > velocity[10]
    assert velocity[2] > 1.5


def test_transcript_keyword_scoring():
    from clipbot.detectors.transcript import DEFAULT_HIGH_KEYWORDS, DEFAULT_MED_KEYWORDS

    # 1. English
    score_en = _keyword_score("Oh my god he's wasted! Insane clutch!", DEFAULT_HIGH_KEYWORDS, DEFAULT_MED_KEYWORDS)
    assert score_en >= 3.0

    # 2. Hindi & Hinglish
    score_hi = _keyword_score("bhai ekdum hard play phod diya", DEFAULT_HIGH_KEYWORDS, DEFAULT_MED_KEYWORDS)
    assert score_hi >= 3.0

    # 3. Malayalam & Manglish
    score_ml = _keyword_score("adipoli machane vera level mass play", DEFAULT_HIGH_KEYWORDS, DEFAULT_MED_KEYWORDS)
    assert score_ml >= 3.0

    # 4. Spanish
    score_es = _keyword_score("dios mio vamos que locura partidaza", DEFAULT_HIGH_KEYWORDS, DEFAULT_MED_KEYWORDS)
    assert score_es >= 3.0

    # 5. Non-hype baseline
    score_zero = _keyword_score("just walking down the street peacefully", DEFAULT_HIGH_KEYWORDS, DEFAULT_MED_KEYWORDS)
    assert score_zero == 0.0


def test_audio_detector_sync_analysis(tmp_path: Path):
    wav_file = tmp_path / "test_audio.wav"
    generate_real_wav(wav_file, duration_sec=3.0, sample_rate=16000, freq=440.0, with_silence=True)

    det = AudioDetector()
    res = det._analyze_audio_sync(wav_file)

    assert "audio_rms" in res
    assert "audio_onset" in res
    assert "audio_vad" in res
    assert "audio_pitch" in res
    assert len(res["audio_rms"]) >= 2
    assert len(res["audio_onset"]) >= 2
    assert len(res["audio_vad"]) >= 2
    assert len(res["audio_pitch"]) >= 2
    assert np.all(res["audio_rms"] >= 0)
    assert np.all(res["audio_onset"] >= 0)
    assert np.all(res["audio_vad"] >= 0)
    assert np.all(res["audio_pitch"] >= 0)


def test_transcript_word_boundary_isolation():
    from clipbot.detectors.transcript import DEFAULT_HIGH_KEYWORDS, DEFAULT_MED_KEYWORDS

    # Substrings inside words should NOT match
    # 'ace' is inside 'peacefully' and 'space'
    assert _keyword_score("peacefully exploring space", DEFAULT_HIGH_KEYWORDS, DEFAULT_MED_KEYWORDS) == 0.0
    # 'poli' (pwoli variant) is inside 'police' or 'policy'
    assert _keyword_score("discussing the new policy", DEFAULT_HIGH_KEYWORDS, DEFAULT_MED_KEYWORDS) == 0.0
    # 'dead' is inside 'deadline'
    assert _keyword_score("working on the project deadline", DEFAULT_HIGH_KEYWORDS, DEFAULT_MED_KEYWORDS) == 0.0

    # But exact standalone words MUST match
    assert _keyword_score("he got an ace", DEFAULT_HIGH_KEYWORDS, DEFAULT_MED_KEYWORDS) > 0.0
    assert _keyword_score("that guy is dead", DEFAULT_HIGH_KEYWORDS, DEFAULT_MED_KEYWORDS) > 0.0
    assert _keyword_score("scene aane bro", DEFAULT_HIGH_KEYWORDS, DEFAULT_MED_KEYWORDS) > 0.0

