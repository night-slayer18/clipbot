"""Wired integration tests for signal combination, peak windowing, and weight tuning."""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import numpy as np
import pytest

from clipbot.core.models import Asset, Clip, ClipStatus
from clipbot.core.repository import insert_clip, set_clip_decision, upsert_asset
from clipbot.scoring.combine import ClipCandidate, _non_max_suppression, _zscore, score_and_window
from clipbot.scoring.tune import tune_weights_from_labels


def test_zscore_handles_nans_and_infinities():
    arr = np.array([1.0, np.nan, 3.0, np.inf, -np.inf, 5.0], dtype=np.float32)
    normalized = _zscore(arr)
    assert not np.isnan(normalized).any()
    assert not np.isinf(normalized).any()


def test_score_and_window_peak_selection():
    # Construct synthetic 100-second signals with an obvious peak at second 40-50
    duration = 100
    audio = np.zeros(duration, dtype=np.float32)
    chat = np.zeros(duration, dtype=np.float32)
    
    # Peak at second 45
    audio[42:48] = 5.0
    chat[43:47] = 8.0

    signals = {
        "audio_rms": audio,
        "chat_velocity": chat,
    }

    candidates = score_and_window(
        signals=signals,
        profile_name="gta_chaos",
        min_duration=25,
        max_duration=45,
        min_gap=30,
        top_n=3,
    )

    assert len(candidates) > 0
    top_cand = candidates[0]
    # The top candidate window should encompass the peak around second 40-50
    assert top_cand.start <= 45 <= top_cand.end
    assert top_cand.end - top_cand.start <= 45


@pytest.mark.asyncio
async def test_adaptive_weight_tuner_with_sqlite(test_db: aiosqlite.Connection, sample_asset: Asset, tmp_path: Path):
    await upsert_asset(test_db, sample_asset)

    # Insert 6 clips with review labels
    for i in range(6):
        is_approved = (i % 2 == 0)
        clip = Clip(
            id=f"tune-clip-{i}",
            asset_id=sample_asset.id,
            overall_score=0.8,
            scores_breakdown={
                "audio_rms": 2.5 if is_approved else 0.2,
                "chat_velocity": 3.0 if is_approved else 0.1,
                "visual_wasted": 1.0 if is_approved else 0.0,
            },
            status=ClipStatus.RENDERED,
        )
        await insert_clip(test_db, clip)
        decision = "approved" if is_approved else "rejected"
        await set_clip_decision(test_db, clip.id, decision)

    db_file = tmp_path / "tune_test.db"
    # Copy in-memory test_db to a file on disk so tune_weights_from_labels can read it
    file_db = await aiosqlite.connect(str(db_file))
    await test_db.backup(file_db)
    await file_db.close()

    tuned_weights = await tune_weights_from_labels(
        db_path=str(db_file),
        profile_name="gta_chaos",
        min_labels=4,
        save=False,
    )

    assert isinstance(tuned_weights, dict)
    assert len(tuned_weights) > 0
    # Weights should sum to approximately 1.0
    assert abs(sum(tuned_weights.values()) - 1.0) < 0.05
