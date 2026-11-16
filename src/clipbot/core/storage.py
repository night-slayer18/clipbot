"""Storage management for ClipBot — cache limits, pruning, and directory cleanup."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from loguru import logger

from clipbot.core.config import get_config


class StorageManager:
    """Manages raw media and rendered clip storage on local disk."""

    def __init__(self, media_dir: str | Path | None = None, clips_dir: str | Path | None = None, max_cache_gb: int | None = None):
        config = get_config()
        self.media_dir = Path(media_dir or config.storage.media_dir)
        self.clips_dir = Path(clips_dir or config.storage.clips_dir)
        self.max_cache_gb = max_cache_gb or config.storage.max_cache_gb

        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

    def get_media_size_bytes(self) -> int:
        """Calculate total bytes occupied by raw media directory."""
        total = 0
        for entry in self.media_dir.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total

    def prune_media_cache(self, keep_latest_n: int = 10) -> int:
        """
        Delete older media files when cache exceeds max_cache_gb.
        Returns the number of deleted files.
        """
        max_bytes = self.max_cache_gb * (1024 ** 3)
        current_bytes = self.get_media_size_bytes()
        deleted_count = 0

        if current_bytes <= max_bytes:
            logger.debug(f"[Storage] Media cache size: {current_bytes / (1024 ** 2):.1f} MB (within limit {self.max_cache_gb} GB)")
            return 0

        logger.info(f"[Storage] Cache ({current_bytes / (1024 ** 3):.2f} GB) exceeds limit ({self.max_cache_gb} GB). Pruning...")

        # Collect files sorted by modification time (oldest first) across all subdirectories
        files = []
        for f in self.media_dir.rglob("*"):
            if f.is_file():
                files.append((f.stat().st_mtime, f.stat().st_size, f))

        files.sort(key=lambda x: x[0])

        # Delete oldest files until cache drops below 80% of max limit
        target_bytes = int(max_bytes * 0.8)
        for _, size, path in files:
            if current_bytes <= target_bytes:
                break
            try:
                path.unlink()
                current_bytes -= size
                deleted_count += 1
                logger.info(f"[Storage] Pruned raw file: {path.name} ({size / (1024 ** 2):.1f} MB)")
            except Exception as exc:
                logger.warning(f"[Storage] Failed to delete {path}: {exc}")

        return deleted_count

    def cleanup_file(self, path: Path | str | None) -> bool:
        """Safely delete a single file."""
        if not path:
            return False
        p = Path(path)
        if p.exists() and p.is_file():
            try:
                p.unlink()
                logger.debug(f"[Storage] Cleaned up temporary file: {p.name}")
                return True
            except Exception as exc:
                logger.warning(f"[Storage] Could not delete {p}: {exc}")
        return False
