"""Weighted signal combiner with Z-score normalization and Non-Max Suppression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clipbot.core.config import load_profile
from clipbot.core.models import Clip


@dataclass
class ClipCandidate:
    start: float
    end: float
    score: float
    breakdown: dict[str, float]


def _zscore(arr: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sigma = clean.mean(), clean.std()
    if sigma < 1e-6:
        return np.zeros_like(clean)
    return (clean - mu) / sigma


def _non_max_suppression(candidates: list[ClipCandidate], min_gap: float) -> list[ClipCandidate]:
    """Remove overlapping candidates, keeping the highest-scored."""
    sorted_cands = sorted(candidates, key=lambda c: -c.score)
    kept: list[ClipCandidate] = []
    for cand in sorted_cands:
        overlap = any(
            not (cand.end + min_gap <= k.start or cand.start >= k.end + min_gap)
            for k in kept
        )
        if not overlap:
            kept.append(cand)
    return sorted(kept, key=lambda c: c.start)


def score_and_window(
    signals: dict[str, np.ndarray],
    profile_name: str,
    min_duration: int = 25,
    max_duration: int = 55,
    min_gap: float = 60.0,
    top_n: int = 20,
) -> list[ClipCandidate]:
    """
    Combine signals using profile weights, find top windows, apply NMS.
    Returns up to top_n non-overlapping ClipCandidate objects.
    """
    profile = load_profile(profile_name)
    weights: dict[str, float] = profile.get("signal_weights", {})
    pre_pad = profile.get("clip", {}).get("pre_pad_seconds", 4)
    post_pad = profile.get("clip", {}).get("post_pad_seconds", 6)

    if not signals:
        return []

    # Align all signals to same length
    min_len = min(len(v) for v in signals.values())
    if min_len < 10:
        return []

    target_dur = min(min_duration, min_len)

    # Normalize each signal to Z-score
    normalized: dict[str, np.ndarray] = {}
    for name, arr in signals.items():
        trimmed = arr[:min_len].astype(np.float32)
        normalized[name] = _zscore(trimmed)

    # Build composite score (weighted sum)
    composite = np.zeros(min_len, dtype=np.float32)
    for signal_name, z in normalized.items():
        # Match by prefix: 'audio_rms' maps to weight key 'audio_rms'
        weight = weights.get(signal_name, 0.0)
        composite += z * weight

    # Sliding window peak detection
    best_starts: list[tuple[float, int]] = []  # (score, start_idx)
    for i in range(0, max(1, min_len - target_dur + 1)):
        window_score = composite[i:i + target_dur].mean()
        best_starts.append((window_score, i))

    # Sort by score, take top candidates
    best_starts.sort(key=lambda x: -x[0])

    candidates: list[ClipCandidate] = []
    for window_score, start_idx in best_starts[:top_n * 5]:
        start = max(0, start_idx - pre_pad)
        end = min(min_len, start_idx + target_dur + post_pad)
        # Clamp duration to max_duration
        if end - start > max_duration:
            end = min(min_len, start + max_duration)

        breakdown = {
            name: float(normalized[name][start_idx:start_idx + target_dur].mean())
            for name in normalized
        }
        candidates.append(ClipCandidate(
            start=float(start),
            end=float(end),
            score=float(window_score),
            breakdown=breakdown,
        ))

    return _non_max_suppression(candidates, min_gap)[:top_n]
