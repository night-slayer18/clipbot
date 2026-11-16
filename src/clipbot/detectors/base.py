"""Base protocol for highlight detectors."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from clipbot.core.models import Asset


@runtime_checkable
class Detector(Protocol):
    """
    A Detector produces a 1Hz signal array for a given asset.
    Returns: dict mapping signal_name -> np.ndarray of shape (duration_seconds,) dtype float32.
    Values are raw (un-normalized) scores; combining.py normalizes them.
    """

    name: str

    async def signals(self, asset: Asset, media_path: Path) -> dict[str, np.ndarray]:
        ...
