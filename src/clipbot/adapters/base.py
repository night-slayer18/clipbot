"""Base protocol for source adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from clipbot.core.models import Asset


@runtime_checkable
class SourceAdapter(Protocol):
    """Every source adapter must implement this interface."""

    name: str

    async def discover(self, limit: int = 50) -> list[Asset]:
        """Discover new assets from the source (clips, VODs)."""
        ...

    async def download(self, asset: Asset, dest_dir: Path) -> Path:
        """Download media to dest_dir. Returns path to file."""
        ...

    async def fetch_chat(self, asset: Asset, dest_dir: Path) -> Path | None:
        """Download chat replay if available. Returns path or None."""
        ...
