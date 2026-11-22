"""Targeted verification tests for the 9 audited engine improvements."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest
import httpx

from clipbot.core.config import get_config
from clipbot.core.models import Asset
from clipbot.detectors.transcript import _get_active_keywords
from clipbot.detectors.visual import VisualDetector
from clipbot.review.app import create_app
from clipbot.scoring.combine import (
    ClipCandidate,
    _find_local_peaks,
    _rolling_zscore,
    _zscore,
    score_and_window,
)
from tests.conftest import generate_real_mp4


def test_rolling_zscore_prevents_intro_outlier_domination():
    """Verify that a loud 30-second screaming intro does not suppress genuine peaks 10 minutes later."""
    duration = 600  # 10 minutes (600s)
    signal = np.full(duration, 0.1, dtype=np.float32)

    # 1. Loud stream intro (seconds 0 to 30)
    signal[0:30] = 10.0

    # 2. Genuine gameplay action peak at minute 5 (second 300 to 315)
    signal[300:315] = 8.0

    global_z = _zscore(signal)
    rolling_z = _rolling_zscore(signal, window_size=120)

    # In global z-score, the loud intro dominates the entire mean and std dev
    # In rolling z-score, the action peak at 300s is evaluated relative to its local 120s baseline
    assert rolling_z[305] > 2.0
    # Intro peak is locally elevated, but after 60s the baseline adjusts smoothly
    assert rolling_z[80] < 1.0


def test_true_peak_picking_yields_multiple_distinct_clips():
    """Verify that multiple high-scoring moments yield distinct clips rather than collapsing to 1."""
    duration = 300  # 5 minutes
    audio = np.zeros(duration, dtype=np.float32)
    onset = np.zeros(duration, dtype=np.float32)
    chat = np.zeros(duration, dtype=np.float32)

    # Peak 1 at second 40
    audio[38:43] = 6.0
    onset[38:43] = 4.0
    chat[39:44] = 7.0

    # Peak 2 at second 120
    audio[118:124] = 8.0
    onset[118:124] = 5.0
    chat[120:125] = 9.0

    # Peak 3 at second 220
    audio[218:223] = 5.5
    onset[218:223] = 3.5
    chat[219:224] = 6.0

    signals = {
        "audio_rms": audio,
        "audio_onset": onset,
        "chat_velocity": chat,
    }

    candidates = score_and_window(
        signals=signals,
        profile_name="gta6_chaos",
        min_duration=15,
        max_duration=35,
        min_gap=30.0,
        top_n=3,
    )

    # Must produce 3 distinct clips around the 3 distinct peaks
    assert len(candidates) == 3
    # Check that candidates surround the 3 peaks
    starts = [c.start for c in candidates]
    ends = [c.end for c in candidates]

    # Verify peak 1 coverage (~40)
    assert any(s <= 40 <= e for s, e in zip(starts, ends))
    # Verify peak 2 coverage (~120)
    assert any(s <= 120 <= e for s, e in zip(starts, ends))
    # Verify peak 3 coverage (~220)
    assert any(s <= 220 <= e for s, e in zip(starts, ends))


def test_dead_chat_signal_weight_renormalization():
    """Verify that when chat_velocity is all zeros, weights dynamically re-normalize without penalty."""
    duration = 120
    audio = np.zeros(duration, dtype=np.float32)
    audio[45:55] = 10.0  # Big audio spike
    onset = np.zeros(duration, dtype=np.float32)
    onset[45:55] = 5.0

    # Chat is completely dead (all zeros)
    chat = np.zeros(duration, dtype=np.float32)
    vad = np.zeros(duration, dtype=np.float32)
    vad[45:60] = 1.0
    pitch = np.zeros(duration, dtype=np.float32)
    pitch[45:60] = 2.5

    signals_with_dead_chat = {
        "audio_rms": audio,
        "audio_onset": onset,
        "audio_vad": vad,
        "audio_pitch": pitch,
        "chat_velocity": chat,
    }

    cands = score_and_window(
        signals=signals_with_dead_chat,
        profile_name="gta6_chaos",
        min_duration=15,
        max_duration=35,
        min_gap=30.0,
        top_n=1,
    )

    assert len(cands) == 1
    # Candidate score must not be penalized to near-zero by the dead chat signal
    assert cands[0].score > 0.8


def test_empty_fallback_does_not_truncate_long_video():
    """Verify that a 60-second empty fallback signal does not truncate a 10-minute video."""
    duration = 600  # 10 minutes
    audio = np.zeros(duration, dtype=np.float32)
    # Peak at minute 8 (second 480)
    audio[475:485] = 8.0
    onset = np.zeros(duration, dtype=np.float32)
    onset[475:485] = 4.0

    # An empty fallback signal of length 60 (as was happening with fallback detectors)
    short_fallback = np.zeros(60, dtype=np.float32)

    signals = {
        "audio_rms": audio,
        "audio_onset": onset,
        "chat_velocity": short_fallback,
    }

    cands = score_and_window(
        signals=signals,
        profile_name="gta6_chaos",
        min_duration=20,
        max_duration=40,
        min_gap=30.0,
        top_n=1,
    )

    assert len(cands) == 1
    # Candidate should successfully detect the peak at second 480, far beyond the 60s mark
    assert cands[0].start <= 480 <= cands[0].end


def test_transcript_profile_keyword_merging():
    """Verify profile-specific keywords (e.g. 'vice city', 'leonida') are merged into active keywords."""
    high_kw, med_kw = _get_active_keywords("gta6_chaos")
    assert "vice city" in high_kw
    assert "gta 6" in high_kw
    assert "leonida" in med_kw


@pytest.mark.asyncio
async def test_review_ui_http_basic_auth(tmp_path: Path, monkeypatch):
    """Verify that review dashboard enforces HTTP Basic Auth when password is set in environment."""
    db_path = tmp_path / "auth_test.db"
    config = get_config()
    config.database.path = str(db_path)

    # 1. When password is set, unauthenticated requests must be rejected (401)
    monkeypatch.setenv("CLIPBOT_REVIEW_PASSWORD", "secret123")
    monkeypatch.setenv("CLIPBOT_REVIEW_USER", "admin")
    monkeypatch.delenv("CLIPBOT_REVIEW_ALLOW_OPEN", raising=False)

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 401

        # 2. Invalid credentials must be rejected (401)
        res_bad = await client.get("/health", auth=("admin", "wrongpassword"))
        assert res_bad.status_code == 401

        # 3. Valid credentials succeed (200)
        res_good = await client.get("/health", auth=("admin", "secret123"))
        assert res_good.status_code == 200
        assert res_good.json()["status"] == "ok"
