"""Base protocol for publishers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class PostResult:
    platform: str
    remote_id: str
    url: str
    success: bool
    error: str = ""


@runtime_checkable
class Publisher(Protocol):
    name: str

    async def upload(self, clip_path: Path, title: str, description: str, tags: list[str]) -> PostResult:
        ...
