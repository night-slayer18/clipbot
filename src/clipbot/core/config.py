"""Configuration loader using Pydantic Settings + YAML files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


CONFIG_DIR = Path(__file__).parents[3] / "config"


class DatabaseConfig(BaseModel):
    path: str = "data/state.db"


class StorageConfig(BaseModel):
    media_dir: str = "data/media"
    clips_dir: str = "data/clips"
    max_cache_gb: int = 50


class PipelineConfig(BaseModel):
    cycle_interval_minutes: int = 120
    max_clips_per_cycle: int = 10
    min_clip_duration: int = 25
    max_clip_duration: int = 60
    min_gap_seconds: int = 60


class ScoringConfig(BaseModel):
    profile: str = "gta_chaos"
    top_n: int = 20


class ReviewConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class PublishConfig(BaseModel):
    dry_run: bool = True
    posts_per_day: dict[str, int] = Field(
        default_factory=lambda: {
            "youtube_shorts": 4,
            "tiktok": 3,
            "instagram_reels": 3,
        }
    )
    spread_hours: list[int] = Field(default_factory=lambda: [9, 12, 17, 20])


class AppConfig(BaseModel):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    publish: PublishConfig = Field(default_factory=PublishConfig)


class ClipBotSettings(BaseSettings):
    """Runtime secrets from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="CLIPBOT_", extra="ignore")

    twitch_client_id: str = ""
    twitch_client_secret: str = ""
    kick_client_id: str = ""
    kick_client_secret: str = ""
    youtube_api_key: str = ""
    youtube_oauth_client_secrets_file: str = ""
    tiktok_client_key: str = ""
    tiktok_client_secret: str = ""
    instagram_access_token: str = ""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    raw = _load_yaml(CONFIG_DIR / "config.yaml")
    return AppConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> ClipBotSettings:
    return ClipBotSettings()


def load_profile(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / "profiles" / f"{name}.yaml"
    data = _load_yaml(path)
    if not data:
        raise FileNotFoundError(f"Profile not found: {path}")
    return data


def load_sources() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "sources.yaml")


def load_accounts() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "accounts.yaml")
