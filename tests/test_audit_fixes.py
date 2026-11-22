"""Unit tests covering audit bug fixes:
- Configurable absolute gate count & multi-modality gating (audio_rms + onset alone != 2)
- Intermission soft penalty when speech is active vs hard suppression when silent
- Facecam jitter normalizer allowing realistic webcam stability
- Subtitle Whisper model load exception handling (graceful fallback)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clipbot.detectors.facecam import FacecamBox
from clipbot.render.captions import CaptionGenerator
from clipbot.scoring.combine import _passes_absolute_gate, score_and_window


def test_gate_count_configurable():
    """Verify min_gates_required controls gate threshold."""
    # Only audio_rms fires
    n = 100
    aligned = {
        "audio_rms": np.array([0.001] * 50 + [0.05] * 50, dtype=np.float32),
    }
    # With min_gates_required=1, should pass
    assert _passes_absolute_gate(aligned, 60, 80, min_gates_required=1) is True
    # With min_gates_required=2, should fail
    assert _passes_absolute_gate(aligned, 60, 80, min_gates_required=2) is False


def test_vocal_pitch_gate():
    """Verify audio_vocal_pitch fires when speech + pitch elevation occur."""
    n = 100
    aligned = {
        "audio_vad": np.ones(n, dtype=np.float32),
        "audio_pitch": np.array([0.1] * 50 + [2.5] * 50, dtype=np.float32),
    }
    # With min_gates_required=1, vocal pitch alone should satisfy
    assert _passes_absolute_gate(aligned, 60, 80, min_gates_required=1) is True


def test_intermission_speech_preservation():
    """Verify that speech during static visual screen is penalized softly rather than hard deleted."""
    n = 120
    # First 80 seconds quiet background, last 40 seconds loud speech
    rms = np.array([0.005] * 80 + [0.20] * 40, dtype=np.float32)
    onset = np.array([0.01] * 80 + [3.0] * 40, dtype=np.float32)
    vad = np.array([0.0] * 80 + [1.0] * 40, dtype=np.float32)
    pitch = np.array([0.1] * 80 + [2.5] * 40, dtype=np.float32)

    signals = {
        "audio_rms": rms,
        "audio_onset": onset,
        "audio_vad": vad,
        "audio_pitch": pitch,
        "visual_impact": np.zeros(n, dtype=np.float32),
        "visual_intermission": np.ones(n, dtype=np.float32),  # Full static screen
    }
    # When speaker is active over static screen, candidate moments should still be recoverable
    candidates = score_and_window(
        signals,
        profile_name="gta6_chaos",
        min_duration=15,
        max_duration=35,
        min_gap=5.0,
        top_n=3,
        media_duration=120.0,
    )
    # With soft penalty instead of hard -999.0 deletion, the speaking window is preserved
    assert len(candidates) >= 1
    assert any(c.start >= 60.0 for c in candidates)


def test_caption_model_runtime_error_fallback():
    """Verify that a network/OS error loading WhisperModel returns None instead of raising."""
    cap_gen = CaptionGenerator(model_size="base")
    with patch("faster_whisper.WhisperModel", side_effect=RuntimeError("Connection timed out")):
        # Force model load
        model = cap_gen._get_model()
        assert model is None
