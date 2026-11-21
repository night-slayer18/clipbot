"""Weighted signal combiner with rolling Z-score normalization, true peak-picking,
dynamic envelope expansion, speech-boundary snapping, and Adaptive Non-Max Suppression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    """Standard global z-score normalization."""
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sigma = clean.mean(), clean.std()
    if sigma < 1e-6:
        return np.zeros_like(clean)
    return (clean - mu) / sigma


def _rolling_zscore(arr: np.ndarray, window_size: int = 120) -> np.ndarray:
    """
    Rolling baseline z-score normalization over a sliding temporal window (default 120s).
    Prevents loud intros or isolated outlier blasts from skewing the entire timeline.
    Floors local standard deviation to avoid noise amplification in quiet periods.
    """
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    n = len(clean)
    if n < 10:
        return np.zeros_like(clean, dtype=np.float32)

    w = min(window_size, n)
    global_std = float(clean.std())
    if w < 10 or global_std < 1e-6:
        return _zscore(clean).astype(np.float32)

    # Box filter via convolution for fast O(N) rolling statistics
    kernel = np.ones(w, dtype=np.float64) / float(w)
    rolling_mean = np.convolve(clean, kernel, mode="same")
    rolling_sq_mean = np.convolve(clean ** 2, kernel, mode="same")
    rolling_var = np.maximum(0.0, rolling_sq_mean - rolling_mean ** 2)
    rolling_std = np.sqrt(rolling_var)

    # Floor rolling_std at 15% of global standard deviation to avoid amplifying background noise
    min_std = max(1e-4, global_std * 0.15)
    safe_std = np.maximum(rolling_std, min_std)

    z = (clean - rolling_mean) / safe_std
    return np.nan_to_num(z, nan=0.0).astype(np.float32)


def _snap_to_vad_pause(
    t: float,
    vad: np.ndarray,
    search_window: float = 3.0,
    is_end: bool = False,
) -> float:
    """Snap candidate boundary timestamp to the nearest speech silence gap (VAD == 0).

    Prevents cutting off streamers mid-word or mid-sentence.
    """
    if len(vad) == 0:
        return t
    t_sec = int(round(t))
    low = max(0, int(t_sec - search_window))
    high = min(len(vad), int(t_sec + search_window + 1))

    # Check if already in a pause
    if 0 <= t_sec < len(vad) and vad[t_sec] < 0.5:
        return float(t_sec)

    pause_indices = [i for i in range(low, high) if vad[i] < 0.5]
    if not pause_indices:
        return t

    if not is_end:
        # Start of clip: prefer silence just before speech starts
        best_idx = min(pause_indices, key=lambda i: (abs(i - t_sec), i > t_sec))
    else:
        # End of clip: prefer silence just after speech completes
        best_idx = min(pause_indices, key=lambda i: (abs(i - t_sec), i < t_sec))

    return float(best_idx)


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


def _find_local_peaks(signal: np.ndarray, min_distance: int, min_score: float) -> list[int]:
    """
    Find distinct local maxima separated by at least min_distance samples.
    Returns peak indices sorted by peak height descending.
    """
    n = len(signal)
    if n < 3:
        return [0] if n > 0 and signal[0] >= min_score else []

    # Smooth signal with a 3-tap box filter to eliminate single-sample jitter
    smoothed = np.convolve(signal, np.ones(3, dtype=np.float32) / 3.0, mode="same")

    # Detect local maxima
    raw_peaks: list[int] = []
    for i in range(1, n - 1):
        if smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            if smoothed[i] >= min_score:
                raw_peaks.append(i)

    # If boundary points are prominent peaks
    if n > 0 and smoothed[0] > smoothed[1] and smoothed[0] >= min_score:
        raw_peaks.append(0)
    if n > 1 and smoothed[-1] > smoothed[-2] and smoothed[-1] >= min_score:
        raw_peaks.append(n - 1)

    if not raw_peaks:
        # If no peak strictly exceeded min_score, check global maximum
        max_idx = int(np.argmax(smoothed))
        if smoothed[max_idx] >= min_score:
            return [max_idx]
        return []

    # Sort peaks descending by smoothed height
    raw_peaks.sort(key=lambda idx: -smoothed[idx])

    # Greedy distance suppression
    selected_peaks: list[int] = []
    for p in raw_peaks:
        if all(abs(p - s) >= min_distance for s in selected_peaks):
            selected_peaks.append(p)

    return selected_peaks


def _expand_envelope(
    peak_idx: int,
    composite: np.ndarray,
    vad: Optional[np.ndarray],
    min_dur: int,
    max_dur: int,
    pre_pad: int,
    post_pad: int,
) -> tuple[float, float]:
    """
    Dynamically expand clip envelope outward from peak_idx until activity subsides
    or hits a natural silence pause, bounded by [min_dur, max_dur].
    """
    total_len = len(composite)
    floor_val = max(0.2, float(composite[peak_idx]) * 0.25)

    # 1. Expand left
    t_start = peak_idx
    max_left = max(0, peak_idx - pre_pad - int(min_dur * 0.4))
    while t_start > max_left:
        # Stop expanding if signal dropped below activity floor and silence detected
        if composite[t_start - 1] < floor_val:
            if vad is None or vad[t_start - 1] < 0.5:
                break
        t_start -= 1

    # 2. Expand right
    t_end = peak_idx + 1
    max_right = min(total_len, peak_idx + post_pad + int(min_dur * 0.6))
    while t_end < max_right:
        if composite[t_end] < floor_val:
            if vad is None or vad[t_end] < 0.5:
                # If we have reached at least min_dur, stop
                if (t_end - t_start) >= min_dur:
                    break
        t_end += 1

    # 3. Ensure minimum duration constraint
    current_dur = t_end - t_start
    if current_dur < min_dur:
        needed = min_dur - current_dur
        # Expand symmetrically
        add_left = min(t_start, int(np.ceil(needed / 2.0)))
        t_start -= add_left
        needed -= add_left
        add_right = min(total_len - t_end, needed)
        t_end += add_right
        # If right expansion still leaves us short, push left further if possible
        if t_end - t_start < min_dur and t_start > 0:
            t_start = max(0, t_end - min_dur)

    # 4. Enforce max duration constraint
    if t_end - t_start > max_dur:
        # Keep peak centered if possible
        half = max_dur / 2.0
        t_start = max(0, int(peak_idx - half))
        t_end = min(total_len, t_start + max_dur)
        if t_end - t_start > max_dur:
            t_end = t_start + max_dur

    start = float(max(0, t_start))
    end = float(min(total_len, t_end))

    # 5. Snap boundaries to natural speech pauses if VAD is present
    if vad is not None and len(vad) > 0:
        snapped_start = _snap_to_vad_pause(start, vad[:total_len], search_window=2.5, is_end=False)
        snapped_end = _snap_to_vad_pause(end, vad[:total_len], search_window=2.5, is_end=True)
        if min_dur <= snapped_end - snapped_start <= max_dur:
            start = snapped_start
            end = snapped_end

    return start, end


def score_and_window(
    signals: dict[str, np.ndarray],
    profile_name: str,
    min_duration: int = 25,
    max_duration: int = 55,
    min_gap: float = 60.0,
    top_n: int = 20,
) -> list[ClipCandidate]:
    """
    Combine signals using profile weights, find top windows, snap to speech pauses, and apply adaptive NMS.
    Returns up to top_n non-overlapping ClipCandidate objects.
    """
    profile = load_profile(profile_name)
    weights: dict[str, float] = profile.get("signal_weights", {})
    clip_cfg = profile.get("clip", {})
    pre_pad = clip_cfg.get("pre_pad_seconds", 4)
    post_pad = clip_cfg.get("post_pad_seconds", 6)
    min_score_threshold = float(profile.get("scoring", {}).get("min_composite_score", 0.40))
    top_n = profile.get("scoring", {}).get("top_n", top_n)

    if not signals:
        return []

    # Fix Duration Truncation Bug:
    # Determine target duration from the longest non-empty signal array,
    # and zero-pad shorter signals (e.g. empty fallback chat/audio) to target length.
    active_lens = [len(v) for v in signals.values() if len(v) > 0 and np.any(v != 0)]
    target_len = max(active_lens) if active_lens else max(len(v) for v in signals.values())
    if target_len < 6:
        return []

    aligned: dict[str, np.ndarray] = {}
    for name, arr in signals.items():
        if len(arr) < target_len:
            padded = np.zeros(target_len, dtype=np.float32)
            padded[:len(arr)] = arr
            aligned[name] = padded
        else:
            aligned[name] = arr[:target_len].astype(np.float32)

    # Dynamic Weight Re-normalization:
    # If a signal (like chat_velocity) is dead/zero-variance, redistribute its weight
    # proportionally across active signals so composite scores are not discounted.
    active_weights: dict[str, float] = {}
    for name, arr in aligned.items():
        if name == "visual_intermission":
            continue
        w = weights.get(name, 0.0)
        if w <= 0.0:
            continue
        if float(np.std(arr)) < 1e-6:
            continue  # Dead signal has zero variance
        active_weights[name] = w

    total_active_w = sum(active_weights.values())
    total_cfg_w = sum(w for k, w in weights.items() if k != "visual_intermission")
    if total_active_w > 1e-6 and total_cfg_w > 0:
        scale = total_cfg_w / total_active_w
        effective_weights = {k: v * scale for k, v in active_weights.items()}
    else:
        effective_weights = weights

    # Adapt target clip duration and NMS suppression gap for shorter media
    if top_n > 1 and target_len < top_n * (min_duration + 15):
        target_dur = max(6, min(min_duration, int(target_len / (top_n * 1.1))))
        effective_max = max(target_dur + 3, min(max_duration, int(target_len / top_n * 1.2)))
        effective_gap = max(1.0, min(10.0, float(target_len - top_n * target_dur) / max(1, top_n)))
        pre_pad = min(pre_pad, max(1, int(target_dur * 0.1)))
        post_pad = min(post_pad, max(1, int(target_dur * 0.1)))
    else:
        target_dur = min(min_duration, target_len)
        effective_max = max_duration
        if min_gap == 60.0 and target_len > 0:
            effective_gap = max(10.0, min(60.0, float(target_len) * 0.05))
        else:
            effective_gap = min_gap

    # Normalize each signal using rolling Z-scores
    normalized: dict[str, np.ndarray] = {}
    for name, arr in aligned.items():
        normalized[name] = _rolling_zscore(arr, window_size=120)

    # Build composite score (weighted sum)
    composite = np.zeros(target_len, dtype=np.float32)
    for signal_name, z in normalized.items():
        if signal_name == "visual_intermission":
            continue
        weight = effective_weights.get(signal_name, 0.0)
        composite += z * weight

    # Zero out / suppress candidate scoring during non-content screens (Starting Soon, BRB, End Credits)
    intermission = aligned.get("visual_intermission", np.zeros(target_len, dtype=np.float32))
    composite[intermission > 0.5] = -999.0

    # True Peak Picking (Local Maxima with Minimum Peak Distance)
    # Replaces naive sliding windows that collapse to 1 single peak copy
    peak_min_dist = max(5, int(effective_gap * 0.5))
    peak_indices = _find_local_peaks(composite, min_distance=peak_min_dist, min_score=min_score_threshold)

    # If no local peaks meet min_score_threshold, fall back to sliding windows if score is acceptable
    if not peak_indices:
        best_starts: list[tuple[float, int]] = []
        for i in range(0, max(1, target_len - target_dur + 1)):
            window_score = float(composite[i:i + target_dur].mean())
            if window_score >= min_score_threshold * 0.6:
                best_starts.append((window_score, i))
        best_starts.sort(key=lambda x: -x[0])
        peak_indices = [idx + target_dur // 2 for _, idx in best_starts[:top_n * 3]]

    # If still no peaks above threshold, video is dull/quiet: return empty (no "best of nothing" clips)
    if not peak_indices:
        # Check if the global maximum is at least slightly positive
        global_max = float(composite.max())
        if global_max > 0.3:
            peak_indices = [int(np.argmax(composite))]
        else:
            return []

    vad = aligned.get("audio_vad")
    candidates: list[ClipCandidate] = []

    # Dynamic Envelope Expansion around each distinct peak
    for p in peak_indices:
        start, end = _expand_envelope(
            peak_idx=p,
            composite=composite,
            vad=vad,
            min_dur=target_dur,
            max_dur=effective_max,
            pre_pad=pre_pad,
            post_pad=post_pad,
        )
        s_idx, e_idx = int(start), int(end)
        if e_idx <= s_idx:
            continue
        window_score = float(composite[s_idx:e_idx].mean())

        breakdown = {
            name: float(normalized[name][s_idx:e_idx].mean())
            for name in normalized
        }
        candidates.append(ClipCandidate(
            start=start,
            end=end,
            score=window_score,
            breakdown=breakdown,
        ))

    # Adaptive Non-Max Suppression
    kept = _non_max_suppression(candidates, effective_gap)

    # If NMS pruned and we have fewer than top_n clips, fill with distinct non-overlapping peaks
    if len(kept) < top_n and target_len >= top_n * 6:
        for cand in sorted(candidates, key=lambda c: -c.score):
            if len(kept) >= top_n:
                break
            overlap = any(
                not (cand.end <= k.start + 1.0 or cand.start >= k.end - 1.0)
                for k in kept
            )
            if not overlap and (cand.end - cand.start) >= max(6.0, target_dur * 0.5):
                kept.append(cand)
        kept.sort(key=lambda c: c.start)

    return kept[:top_n]
