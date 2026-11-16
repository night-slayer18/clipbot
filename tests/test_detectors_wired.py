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
from clipbot.detectors.facecam import FacecamBox, FacecamDetector
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
    assert len(signals["visual_impact"]) >= 3
    assert len(signals["visual_scene"]) >= 3
    assert len(signals["visual_motion"]) >= 3


def test_facecam_crop_even_parity():
    # Test that FacecamBox enforces even dimensions for FFmpeg yuv420p
    box = FacecamBox(x=10, y=20, w=345, h=193, confidence=0.85)
    crop_str = box.to_ffmpeg_crop()
    # w:h:x:y -> 344:192:10:20
    parts = [int(p) for p in crop_str.split(":")]
    assert parts[0] % 2 == 0  # Width is even
    assert parts[1] % 2 == 0  # Height is even
    assert parts[0] == 344
    assert parts[1] == 192


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
    high_kw = ["wasted", "goal", "dunk", "mar gaya"]
    med_kw = ["bro", "what the", "chalo"]

    score_high = _keyword_score("Oh my god he's wasted!", high_kw, med_kw)
    assert score_high >= 1.5

    score_med = _keyword_score("what the bro", high_kw, med_kw)
    assert score_med >= 2.0  # 1.0 (what the) + 1.0 (bro)

    score_zero = _keyword_score("just walking down the street", high_kw, med_kw)
    assert score_zero == 0.0


def test_audio_detector_sync_analysis(tmp_path: Path):
    wav_file = tmp_path / "test_audio.wav"
    generate_real_wav(wav_file, duration_sec=3.0, sample_rate=16000, freq=440.0, with_silence=True)

    det = AudioDetector()
    rms, onset = det._analyze_audio_sync(wav_file)

    assert len(rms) >= 2
    assert len(onset) >= 2
    assert np.all(rms >= 0)
    assert np.all(onset >= 0)
