"""Adaptive scoring weight tuner — refits signal weights from approve/reject review labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np
import yaml
from loguru import logger

from clipbot.core.config import CONFIG_DIR, get_config, load_profile


async def tune_weights_from_labels(
    db_path: str = "data/state.db",
    profile_name: str = "gta_chaos",
    min_labels: int = 5,
    save: bool = True,
) -> dict[str, float]:
    """
    Read human review decisions (labels) from DB, correlate signal breakdowns
    with approved vs rejected decisions, and optimize signal weights.
    """
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            """
            SELECT l.decision, c.scores_breakdown
            FROM labels l
            JOIN clips c ON l.clip_id = c.id
            WHERE c.scores_breakdown IS NOT NULL
            """
        ) as cur:
            rows = await cur.fetchall()

    if len(rows) < min_labels:
        logger.warning(
            f"[Tune] Only {len(rows)} labelled clips found (need at least {min_labels}). Weights unchanged."
        )
        return load_profile(profile_name).get("signal_weights", {})

    logger.info(f"[Tune] Tuning weights using {len(rows)} human review decisions")

    # Parse dataset
    labels: list[int] = []
    sample_breakdowns: list[dict[str, float]] = []
    all_signals: set[str] = set()

    for decision, breakdown_raw in rows:
        try:
            breakdown = json.loads(breakdown_raw)
            if not isinstance(breakdown, dict):
                continue
            is_approved = 1 if decision == "approved" else 0
            labels.append(is_approved)
            parsed_bd = {k: float(v) for k, v in breakdown.items() if isinstance(v, (int, float))}
            sample_breakdowns.append(parsed_bd)
            all_signals.update(parsed_bd.keys())
        except Exception:
            continue

    if not labels or not all_signals:
        return load_profile(profile_name).get("signal_weights", {})

    y = np.array(labels, dtype=np.float32)

    # Build dense feature matrix (guaranteeing exact same length for all signals)
    features: dict[str, list[float]] = {sig: [] for sig in all_signals}
    for bd in sample_breakdowns:
        for sig in all_signals:
            features[sig].append(bd.get(sig, 0.0))

    # Compute correlation with positive approvals
    correlations: dict[str, float] = {}
    for sig_name, vals in features.items():
        x = np.array(vals, dtype=np.float32)
        std_x = np.std(x)
        std_y = np.std(y)
        if std_x > 1e-5 and std_y > 1e-5:
            corr = float(np.corrcoef(x, y)[0, 1])
            # ReLU on correlation (we want positive predictors)
            correlations[sig_name] = max(0.05, corr if not np.isnan(corr) else 0.1)
        else:
            correlations[sig_name] = 0.1

    # Normalize weights to sum to 1.0
    total = sum(correlations.values())
    new_weights = {k: round(v / total, 3) for k, v in correlations.items()}

    logger.info(f"[Tune] Optimized weights for profile '{profile_name}': {new_weights}")

    if save:
        profile_path = CONFIG_DIR / "profiles" / f"{profile_name}.yaml"
        if profile_path.exists():
            with profile_path.open() as f:
                data = yaml.safe_load(f) or {}
            data["signal_weights"] = new_weights
            with profile_path.open("w") as f:
                yaml.safe_dump(data, f)
            logger.success(f"[Tune] Updated profile file: {profile_path}")

    return new_weights
