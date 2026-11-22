"""Weighted signal combiner with rolling Z-score normalization, true peak-picking,
dynamic envelope expansion, speech-boundary snapping, and Adaptive Non-Max Suppression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

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


def _rolling_zscore(arr: np.ndarray, window_size: int = 600) -> np.ndarray:
    """
    Rolling baseline z-score using median and MAD over a 10-minute window (default 600s).
    Uses nearest-edge padding to prevent boundary distortion at stream edges.
    """
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    n = len(clean)
    if n < 10:
        return np.zeros_like(clean, dtype=np.float32)

    w = min(window_size, n)
    global_std = float(clean.std())
    if w < 10 or global_std < 1e-6:
        return _zscore(clean).astype(np.float32)

    # Nearest-edge padding to eliminate edge-drop distortion
    pad = w // 2
    padded = np.pad(clean, (pad, pad), mode="edge")

    kernel = np.ones(w, dtype=np.float64) / float(w)
    rolling_mean = np.convolve(padded, kernel, mode="valid")[:n]
    rolling_sq_mean = np.convolve(padded ** 2, kernel, mode="valid")[:n]
    rolling_var = np.maximum(0.0, rolling_sq_mean - rolling_mean ** 2)
    rolling_std = np.sqrt(rolling_var)

    global_mad = float(np.median(np.abs(clean - np.median(clean)))) or 1e-4
    min_std = max(1e-4, global_mad * 1.4826 * 0.15)
    safe_std = np.maximum(rolling_std, min_std)

    z = (clean - rolling_mean) / safe_std
    return np.nan_to_num(z, nan=0.0).astype(np.float32)


def _snap_to_vad_pause(
    t: float,
    vad: np.ndarray,
    search_window: float = 3.0,
    is_end: bool = False,
) -> float:
    """Snap candidate boundary timestamp to the nearest speech silence gap (VAD == 0)."""
    if len(vad) == 0:
        return t
    t_sec = int(round(t))
    low = max(0, int(t_sec - search_window))
    high = min(len(vad), int(t_sec + search_window + 1))

    if 0 <= t_sec < len(vad) and vad[t_sec] < 0.5:
        return float(t_sec)

    pause_indices = [i for i in range(low, high) if vad[i] < 0.5]
    if not pause_indices:
        return t

    if not is_end:
        best_idx = min(pause_indices, key=lambda i: (abs(i - t_sec), i > t_sec))
    else:
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
    Excludes boundary samples to prevent edge false-positives.
    """
    n = len(signal)
    if n < 3:
        return [0] if n > 0 and signal[0] >= min_score else []

    # Smooth with nearest-edge padding
    padded_s = np.pad(signal, (1, 1), mode="edge")
    smoothed = np.convolve(padded_s, np.ones(3, dtype=np.float32) / 3.0, mode="valid")[:n]

    edge_guard = max(1, min(10, min_distance))

    raw_peaks: list[int] = []
    for i in range(max(1, edge_guard), min(n - 1, n - edge_guard)):
        if smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            if smoothed[i] >= min_score:
                raw_peaks.append(i)

    if not raw_peaks:
        interior = smoothed[edge_guard: n - edge_guard]
        if len(interior) == 0:
            interior = smoothed
        max_idx = int(np.argmax(interior)) + (edge_guard if len(interior) < n else 0)
        if smoothed[max_idx] >= min_score:
            return [max_idx]
        return []

    raw_peaks.sort(key=lambda idx: -smoothed[idx])

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
    """Dynamically expand clip envelope outward from peak_idx bounded by [min_dur, max_dur]."""
    total_len = len(composite)
    floor_val = max(0.2, float(composite[peak_idx]) * 0.25)

    # Expand left using max_dur budget
    t_start = peak_idx
    max_left = max(0, peak_idx - max_dur // 2)
    while t_start > max_left:
        if composite[t_start - 1] <= 0.0:
            break
        if composite[t_start - 1] < floor_val:
            if vad is None or vad[t_start - 1] < 0.5:
                break
        t_start -= 1

    # Expand right using max_dur budget
    t_end = peak_idx + 1
    max_right = min(total_len, peak_idx + max_dur - (peak_idx - t_start) + 1)
    while t_end < max_right:
        if composite[t_end] <= 0.0:
            break
        if composite[t_end] < floor_val:
            if vad is None or vad[t_end] < 0.5:
                if (t_end - t_start) >= min_dur:
                    break
        t_end += 1

    # Ensure minimum duration constraint
    current_dur = t_end - t_start
    if current_dur < min_dur:
        needed = min_dur - current_dur
        # Prefer expanding rightward into positive activity before encroaching leftward
        add_right = min(total_len - t_end, needed)
        t_end += add_right
        needed -= add_right
        if needed > 0 and t_start > 0:
            add_left = min(t_start, needed)
            t_start -= add_left

    # Enforce max duration constraint
    if t_end - t_start > max_dur:
        t_end = t_start + max_dur

    start = float(max(0, t_start))
    end = float(min(total_len, t_end))

    # Snap boundaries to natural speech pauses if VAD is present without snapping into dead air
    if vad is not None and len(vad) > 0:
        snapped_start = _snap_to_vad_pause(start, vad[:total_len], search_window=2.5, is_end=False)
        snapped_end = _snap_to_vad_pause(end, vad[:total_len], search_window=2.5, is_end=True)
        if composite[int(snapped_start)] > 0.0 and min_dur <= snapped_end - snapped_start <= max_dur:
            start = snapped_start
            end = snapped_end

    return start, end


def _passes_absolute_gate(
    aligned: dict[str, np.ndarray],
    s_idx: int,
    e_idx: int,
    min_gates_required: int = 2,
) -> bool:
    """
    Absolute multi-signal gate to reject windows with no genuine activity.
    Requires at least min_gates_required distinct signals to fire before
    a candidate is accepted. Logs which gates passed and which failed
    for every rejected window to aid debugging.
    """
    fired: list[str] = []
    missed: list[str] = []

    rms = aligned.get("audio_rms")
    if rms is not None and len(rms) > e_idx:
        med = float(np.median(rms))
        std = float(np.std(rms))
        w_mean = float(rms[s_idx:e_idx].mean())
        if std > 0.005 and w_mean > max(0.01, med * 1.5):
            fired.append("audio_rms")
        else:
            missed.append(f"audio_rms(w_mean={w_mean:.4f},med={med:.4f},std={std:.4f})")

    onset = aligned.get("audio_onset")
    if onset is not None and len(onset) > e_idx:
        med = float(np.median(onset))
        w_max = float(onset[s_idx:e_idx].max())
        if w_max > max(0.3, med * 2.0):
            fired.append("audio_onset")
        else:
            missed.append(f"audio_onset(w_max={w_max:.4f},med={med:.4f})")

    impact = aligned.get("visual_impact")
    if impact is not None and len(impact) > e_idx:
        v = float(impact[s_idx:e_idx].max())
        rms = aligned.get("audio_rms")
        vad = aligned.get("audio_vad")
        # Visual impact only counts towards absolute gate if accompanied by audible reaction
        has_sound = False
        if rms is not None and len(rms) > e_idx:
            has_sound = float(rms[s_idx:e_idx].mean()) > 0.015
        if vad is not None and len(vad) > e_idx:
            has_sound = has_sound or (float(vad[s_idx:e_idx].mean()) > 0.2)
        if v > 0.5 and has_sound:
            fired.append("visual_impact")
        else:
            missed.append(f"visual_impact(max={v:.4f},audible={has_sound})")

    scene = aligned.get("visual_scene")
    if scene is not None and len(scene) > e_idx:
        med = float(np.median(scene))
        v = float(scene[s_idx:e_idx].mean())
        if v > max(0.05, med * 1.8):
            fired.append("visual_scene")
        else:
            missed.append(f"visual_scene(mean={v:.4f},med={med:.4f})")

    chat = aligned.get("chat_velocity")
    if chat is not None and len(chat) > e_idx:
        med = float(np.median(chat))
        w_max = float(chat[s_idx:e_idx].max())
        if w_max > max(1.0, med * 2.0):
            fired.append("chat_velocity")
        else:
            missed.append(f"chat_velocity(w_max={w_max:.4f},med={med:.4f})")

    hype = aligned.get("transcript_hype")
    if hype is not None and len(hype) > e_idx:
        v = float(hype[s_idx:e_idx].max())
        if v > 0.0:
            fired.append("transcript_hype")
        else:
            missed.append(f"transcript_hype(max={v:.4f})")

    # Vocal pitch excitation: confirmed speech (VAD > 0.35) with elevated pitch (screaming, shouting, hype)
    vad = aligned.get("audio_vad")
    pitch = aligned.get("audio_pitch")
    if vad is not None and pitch is not None and len(vad) > e_idx and len(pitch) > e_idx:
        v_mean = float(vad[s_idx:e_idx].mean())
        p_max = float(pitch[s_idx:e_idx].max())
        med_p = float(np.median(pitch))
        if v_mean > 0.35 and p_max > max(1.0, med_p * 1.8):
            fired.append("audio_vocal_pitch")
        else:
            missed.append(f"audio_vocal_pitch(vad={v_mean:.2f},p_max={p_max:.2f})")

    passed = len(fired) >= min_gates_required
    if not passed:
        logger.debug(
            f"[Gate] REJECT window [{s_idx}:{e_idx}] — fired={fired} (need {min_gates_required}) missed={missed}"
        )
    return passed


def score_and_window(
    signals: dict[str, np.ndarray],
    profile_name: str,
    min_duration: int = 25,
    max_duration: int = 55,
    min_gap: float = -1.0,
    top_n: int = 20,
    media_duration: Optional[float] = None,
) -> list[ClipCandidate]:
    """
    Combine signals using profile weights, find top windows, snap to speech pauses, and apply adaptive NMS.
    Returns up to top_n non-overlapping ClipCandidate objects in chronological order.
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

    active_lens = [len(v) for v in signals.values() if len(v) > 0 and np.any(v != 0)]
    raw_target = max(active_lens) if active_lens else max(len(v) for v in signals.values())

    if media_duration is not None and media_duration > 0:
        target_len = min(raw_target, int(np.ceil(media_duration)))
    else:
        target_len = raw_target

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

    # Re-normalize weights if some signals have zero variance
    active_weights: dict[str, float] = {}
    for name, arr in aligned.items():
        if name == "visual_intermission":
            continue
        w = weights.get(name, 0.0)
        if w <= 0.0:
            continue
        if float(np.std(arr)) < 1e-6:
            continue
        active_weights[name] = w

    total_active_w = sum(active_weights.values())
    total_cfg_w = sum(w for k, w in weights.items() if k != "visual_intermission")
    if total_active_w > 1e-6 and total_cfg_w > 0:
        scale = total_cfg_w / total_active_w
        effective_weights = {k: v * scale for k, v in active_weights.items()}
    else:
        effective_weights = weights

    # Adapt duration and gap parameters
    if top_n > 1 and target_len < top_n * (min_duration + 15):
        target_dur = max(6, min(min_duration, int(target_len / (top_n * 1.1))))
        effective_max = max(target_dur + 3, min(max_duration, int(target_len / top_n * 1.2)))
        effective_gap = max(1.0, min(10.0, float(target_len - top_n * target_dur) / max(1, top_n)))
        pre_pad = min(pre_pad, max(1, int(target_dur * 0.1)))
        post_pad = min(post_pad, max(1, int(target_dur * 0.1)))
    else:
        target_dur = min(min_duration, target_len)
        effective_max = max_duration
        if min_gap < 0 and target_len > 0:
            effective_gap = max(10.0, min(25.0, float(target_len) * 0.02))
        else:
            effective_gap = max(0.0, min_gap)

    normalized: dict[str, np.ndarray] = {}
    rms_arr = aligned.get("audio_rms")
    vad_arr = aligned.get("audio_vad")
    for name, arr in aligned.items():
        if name == "visual_impact":
            # Sparse 0/1 pulses: use bounded raw signal gated by audible sound instead of
            # rolling z-score (which inflates sparse 0/1 pulses to +3.0 to +4.5 std deviations).
            raw_imp = np.clip(arr, 0.0, 1.0)
            if rms_arr is not None or vad_arr is not None:
                audible_mask = np.zeros(target_len, dtype=bool)
                if rms_arr is not None:
                    med_rms = float(np.median(rms_arr)) if len(rms_arr) > 0 else 0.01
                    audible_mask |= (rms_arr > max(0.015, med_rms * 1.2))
                if vad_arr is not None:
                    audible_mask |= (vad_arr > 0.3)
                normalized[name] = raw_imp * audible_mask.astype(np.float32)
            else:
                normalized[name] = raw_imp
        else:
            normalized[name] = _rolling_zscore(arr, window_size=600)

    composite = np.zeros(target_len, dtype=np.float32)
    for signal_name, z in normalized.items():
        if signal_name == "visual_intermission":
            continue
        weight = effective_weights.get(signal_name, 0.0)
        composite += z * weight

    intermission = aligned.get("visual_intermission", np.zeros(target_len, dtype=np.float32))
    vad = aligned.get("audio_vad")
    if vad is not None and len(vad) == target_len:
        # Hard suppress static screens only when nobody is speaking (BRB, idle menu, black screen).
        # Use -3.0 instead of -999.0 so a momentary speech pause does not ruin the window mean by -40 points.
        hard_intermission = (intermission > 0.5) & (vad < 0.5)
        soft_intermission = (intermission > 0.5) & (vad >= 0.5)
        composite[hard_intermission] = -3.0
        composite[soft_intermission] -= 0.5
    else:
        composite[intermission > 0.5] = -3.0

    peak_min_dist = max(5, int(effective_gap * 0.5))
    peak_indices = _find_local_peaks(composite, min_distance=peak_min_dist, min_score=min_score_threshold)

    if not peak_indices:
        best_starts: list[tuple[float, int]] = []
        for i in range(0, max(1, target_len - target_dur + 1)):
            window_score = float(composite[i:i + target_dur].mean())
            if window_score >= min_score_threshold * 0.6:
                best_starts.append((window_score, i))
        best_starts.sort(key=lambda x: -x[0])
        peak_indices = [idx + target_dur // 2 for _, idx in best_starts[:top_n * 3]]

    if not peak_indices:
        global_max = float(composite.max())
        if global_max > 0.3:
            peak_indices = [int(np.argmax(composite))]
        else:
            return []

    vad = aligned.get("audio_vad")
    candidates: list[ClipCandidate] = []

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

        min_gates = int(profile.get("scoring", {}).get("min_gates_required", 2))
        if not _passes_absolute_gate(aligned, s_idx, min(e_idx, target_len), min_gates_required=min_gates):
            continue

        window_mean = float(composite[s_idx:e_idx].mean())
        peak_score = float(composite[p])
        # Hybrid ranking: 50% peak strength + 50% window mean to preserve short viral bursts
        window_score = 0.5 * peak_score + 0.5 * window_mean
        if window_score <= 0.0:
            continue

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

    kept = _non_max_suppression(candidates, effective_gap)

    if len(kept) < top_n and target_len >= top_n * 6:
        for cand in sorted(candidates, key=lambda c: -c.score):
            if len(kept) >= top_n:
                break
            overlap = any(
                not (cand.end + effective_gap <= k.start or cand.start >= k.end + effective_gap)
                for k in kept
            )
            if not overlap and (cand.end - cand.start) >= max(6.0, target_dur * 0.5):
                kept.append(cand)
        kept.sort(key=lambda c: c.start)

    # Sort by score descending, truncate to top_n, then re-sort chronologically
    kept.sort(key=lambda c: -c.score)
    kept = kept[:top_n]
    kept.sort(key=lambda c: c.start)

    return kept
