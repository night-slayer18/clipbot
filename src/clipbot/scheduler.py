"""Background scheduler using APScheduler — runs periodic cycles, publishing, and cleanup."""

from __future__ import annotations

import asyncio
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from filelock import FileLock, Timeout
from loguru import logger

from clipbot.core.config import get_config
from clipbot.core.storage import StorageManager
from clipbot.pipeline import run_cycle
from clipbot.publish.manager import PublishManager


LOCK_FILE = Path("data/clipbot.lock")


class ClipBotScheduler:
    """Orchestrates periodic pipeline runs, publishing queues, and disk cleanup."""

    def __init__(self, profile_name: str = "gta_chaos") -> None:
        self.profile_name = profile_name
        self.config = get_config()
        self.scheduler = AsyncIOScheduler()
        self.publish_mgr = PublishManager(self.config.database.path)
        self.storage = StorageManager()

        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(LOCK_FILE, timeout=1)

    async def _scheduled_pipeline_job(self) -> None:
        try:
            with self.lock:
                logger.info("[Scheduler] Starting scheduled pipeline cycle...")
                await run_cycle(profile_name=self.profile_name)
        except Timeout:
            logger.warning("[Scheduler] Cycle already running; skipping this tick.")
        except Exception as exc:
            logger.error(f"[Scheduler] Pipeline cycle failed: {exc}")

    async def _scheduled_publish_job(self) -> None:
        try:
            logger.info("[Scheduler] Checking publish queue for approved clips...")
            await self.publish_mgr.dispatch_approved_clips()
        except Exception as exc:
            logger.error(f"[Scheduler] Publish job failed: {exc}")

    async def _scheduled_cleanup_job(self) -> None:
        try:
            logger.info("[Scheduler] Running daily storage cache pruning...")
            deleted = self.storage.prune_media_cache()
            logger.info(f"[Scheduler] Pruned {deleted} old files.")
        except Exception as exc:
            logger.error(f"[Scheduler] Cleanup failed: {exc}")

    def start(self) -> None:
        """Start the background scheduler."""
        interval_min = self.config.pipeline.cycle_interval_minutes

        # Pipeline cycle every N minutes
        self.scheduler.add_job(
            self._scheduled_pipeline_job,
            "interval",
            minutes=interval_min,
            id="pipeline_cycle",
            replace_existing=True,
        )

        # Publish check every 30 minutes
        self.scheduler.add_job(
            self._scheduled_publish_job,
            "interval",
            minutes=30,
            id="publish_dispatch",
            replace_existing=True,
        )

        # Storage cleanup daily at 03:00
        self.scheduler.add_job(
            self._scheduled_cleanup_job,
            "cron",
            hour=3,
            minute=0,
            id="storage_cleanup",
            replace_existing=True,
        )

        self.scheduler.start()
        logger.success(f"[Scheduler] Started! Cycle every {interval_min}m, publish check every 30m.")


async def run_daemon(profile: str = "gta_chaos") -> None:
    scheduler = ClipBotScheduler(profile_name=profile)
    scheduler.start()
    try:
        # Run first cycle immediately
        await scheduler._scheduled_pipeline_job()
        while True:
            await asyncio.sleep(3600)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("[Scheduler] Shutting down scheduler daemon...")
        scheduler.scheduler.shutdown(wait=False)
        logger.success("[Scheduler] Stopped gracefully.")
