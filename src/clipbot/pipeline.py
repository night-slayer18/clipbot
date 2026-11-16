"""Master pipeline orchestrator — end-to-end discover -> download -> score -> render cycle."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import aiosqlite
from loguru import logger

from clipbot.adapters.kick import KickAdapter
from clipbot.adapters.twitch import TwitchAdapter
from clipbot.adapters.youtube import YouTubeAdapter
from clipbot.core.config import get_config, load_profile
from clipbot.core.db import get_db, init_db
from clipbot.core.models import Asset, AssetStatus, Clip, ClipStatus
from clipbot.core.repository import (
    get_asset_by_external, get_signals, has_signals, insert_clip, update_asset_status,
    upsert_asset, upsert_signal,
)
from clipbot.core.storage import StorageManager
from clipbot.detectors.audio import AudioDetector
from clipbot.detectors.chat import ChatDetector
from clipbot.detectors.transcript import TranscriptDetector
from clipbot.detectors.visual import VisualDetector
from clipbot.meta.titles import generate_description, generate_tags, generate_title
from clipbot.render.compositor import VideoCompositor
from clipbot.scoring.combine import score_and_window


async def run_cycle(profile_name: str = "gta_chaos", auto_render: bool = True) -> list[Clip]:
    """
    Run one full pipeline cycle:
      1. Discover top viral moments from Twitch, Kick, YouTube (Gaming, Sports, Movies, IRL)
      2. Download raw media
      3. Run multi-modal detectors (Audio, Chat, Visual, Transcript)
      4. Score & window highlights using active profile weights
      5. Render 9:16 vertical shorts with smart framing and burned karaoke subtitles
      6. Queue clips in SQLite for human review
    """
    config = get_config()
    db_path = config.database.path
    await init_db(db_path)

    storage = StorageManager(
        media_dir=config.storage.media_dir,
        clips_dir=config.storage.clips_dir,
        max_cache_gb=config.storage.max_cache_gb,
    )

    logger.info(f"[Pipeline] Starting cycle with profile '{profile_name}'")

    # 1. Discover assets from all sources
    adapters = [TwitchAdapter(), KickAdapter(), YouTubeAdapter()]
    all_discovered: list[Asset] = []

    for adapter in adapters:
        try:
            found = await adapter.discover(limit=30)
            all_discovered.extend(found)
            logger.info(f"[Pipeline] {adapter.name}: discovered {len(found)} assets")
        except Exception as exc:
            logger.error(f"[Pipeline] {adapter.name} discovery error: {exc}")

    logger.info(f"[Pipeline] Total discovered candidates: {len(all_discovered)}")
    if not all_discovered:
        logger.warning("[Pipeline] No candidates discovered in this cycle")
        return []

    # Initialize detectors & compositor
    audio_det = AudioDetector()
    chat_det = ChatDetector()
    visual_det = VisualDetector()
    transcript_det = TranscriptDetector(model_size="base")
    compositor = VideoCompositor(profile_name=profile_name)

    db = await get_db(db_path)
    processed_clips: list[Clip] = []

    try:
        # 2. Process assets up to cycle limit
        limit = config.pipeline.max_clips_per_cycle
        for asset in all_discovered[:limit]:
            # Skip if asset was already fully processed in a previous cycle
            existing = await get_asset_by_external(db, asset.source.value, asset.external_id)
            if existing and existing.status in (AssetStatus.READY, AssetStatus.SCORED, AssetStatus.ARCHIVED):
                if existing.local_path and existing.local_path.exists():
                    logger.info(f"[Pipeline] Skipping already processed asset: {asset.title[:50]}")
                    continue

            logger.info(f"\n[Pipeline] === Processing Asset: {asset.title[:60]} ({asset.source.value}) ===")
            await upsert_asset(db, asset)

            # Download media
            adapter = next((a for a in adapters if a.name == asset.source.value), None)
            if not adapter:
                continue

            try:
                asset.local_path = await adapter.download(asset, storage.media_dir)
                asset.status = AssetStatus.READY
                await update_asset_status(db, asset.id, AssetStatus.READY)
            except Exception as exc:
                logger.error(f"[Pipeline] Download failed for {asset.external_id}: {exc}")
                await update_asset_status(db, asset.id, AssetStatus.ERROR)
                continue

            # Check if signals already cached
            signals = await get_signals(db, asset.id)
            if not signals:
                logger.info(f"[Pipeline] Running detector suite on {asset.local_path.name}")
                # Run detectors
                try:
                    audio_sig = await audio_det.signals(asset, asset.local_path)
                    signals.update(audio_sig)
                except Exception as exc:
                    logger.warning(f"[Pipeline] Audio detector error: {exc}")

                try:
                    chat_sig = await chat_det.signals(asset, asset.local_path)
                    signals.update(chat_sig)
                except Exception as exc:
                    logger.warning(f"[Pipeline] Chat detector error: {exc}")

                try:
                    visual_sig = await visual_det.signals(asset, asset.local_path)
                    signals.update(visual_sig)
                except Exception as exc:
                    logger.warning(f"[Pipeline] Visual detector error: {exc}")

                try:
                    trans_sig = await transcript_det.signals(asset, asset.local_path)
                    signals.update(trans_sig)
                except Exception as exc:
                    logger.warning(f"[Pipeline] Transcript detector error: {exc}")

                # Cache signals in DB
                for name, arr in signals.items():
                    await upsert_signal(db, asset.id, name, arr)

            # 3. Score & Window candidate highlights
            candidates = score_and_window(
                signals=signals,
                profile_name=profile_name,
                min_duration=config.pipeline.min_clip_duration,
                max_duration=config.pipeline.max_clip_duration,
                min_gap=float(config.pipeline.min_gap_seconds),
                top_n=3,  # Top moments per asset
            )

            # If asset is already a short clip (<60s) and no window found, treat full asset as candidate
            if not candidates and asset.duration and asset.duration <= 60:
                logger.info(f"[Pipeline] Short asset detected; treating whole duration as highlight")
                from clipbot.scoring.combine import ClipCandidate
                candidates = [ClipCandidate(start=0.0, end=asset.duration, score=1.0, breakdown={})]

            logger.info(f"[Pipeline] Found {len(candidates)} candidate moments for {asset.external_id}")

            # 4. Render and queue each candidate
            for cand in candidates:
                clip = Clip(
                    asset_id=asset.id,
                    start_time=cand.start,
                    end_time=cand.end,
                    overall_score=cand.score,
                    scores_breakdown=cand.breakdown,
                    status=ClipStatus.CANDIDATE,
                )

                # Generate title, description, and tags dynamically based on profile
                clip.title = generate_title(asset, clip, profile_name=profile_name)
                clip.tags = generate_tags(category=profile_name)
                clip.description = generate_description(asset, clip.tags, profile_name=profile_name)

                if auto_render:
                    out_path = storage.clips_dir / f"clip_{clip.id}_9x16.mp4"
                    try:
                        rendered_path = await compositor.render_short(
                            source_media=asset.local_path,
                            start_time=clip.start_time,
                            end_time=clip.end_time,
                            output_path=out_path,
                        )
                        clip.rendered_clip_path = rendered_path
                        clip.status = ClipStatus.RENDERED
                    except Exception as exc:
                        logger.error(f"[Pipeline] Render failed for candidate {clip.id}: {exc}")

                await insert_clip(db, clip)
                processed_clips.append(clip)
                logger.success(f"[Pipeline] Queued Clip: '{clip.title}' (Score: {clip.overall_score:.2f})")

        # 5. Prune cache if needed
        try:
            storage.prune_media_cache()
        except Exception as exc:
            logger.warning(f"[Pipeline] Storage cache prune notice: {exc}")

    finally:
        await db.close()

    logger.success(f"[Pipeline] Cycle complete! {len(processed_clips)} clips generated.")
    return processed_clips


async def process_url_pipeline(
    url: str,
    profile_name: str = "general_viral",
    max_clips: int = 3,
) -> list[Clip]:
    """Execute the full highlights pipeline on a single URL, rendering up to max_clips."""
    config = get_config()
    db_path = config.database.path
    await init_db(db_path)

    storage = StorageManager()
    logger.info(f"[ClipPipeline] Processing highlights on {url} (max {max_clips} clips)")

    from clipbot.adapters.downloader import MediaDownloader
    temp_media = storage.media_dir / "test_input.mp4"
    downloaded = await MediaDownloader.download_media(url, temp_media)

    asset = Asset(
        external_id="url_stream_clip",
        creator="ClipBot Stream",
        title="Stream Highlights",
        url=url,
        local_path=downloaded,
        status=AssetStatus.READY,
    )

    db = await get_db(db_path)
    await upsert_asset(db, asset)

    # Extract signals
    audio_det = AudioDetector()
    visual_det = VisualDetector()
    transcript_det = TranscriptDetector(model_size="base")

    logger.info("[ClipPipeline] Running multi-modal detectors...")
    signals = {}
    signals.update(await audio_det.signals(asset, downloaded))
    signals.update(await visual_det.signals(asset, downloaded))
    signals.update(await transcript_det.signals(asset, downloaded))

    # Save signals
    for name, arr in signals.items():
        await upsert_signal(db, asset.id, name, arr)

    # Window candidates (ignoring intermission and credits)
    candidates = score_and_window(signals, profile_name=profile_name, top_n=max_clips)
    if not candidates:
        logger.warning("[ClipPipeline] No peaks found; using opening segment")
        sig_len = float(len(next(iter(signals.values())))) if signals else 30.0
        from clipbot.scoring.combine import ClipCandidate
        candidates = [ClipCandidate(start=0.0, end=min(30.0, sig_len), score=1.0, breakdown={})]

    logger.info(f"[ClipPipeline] Selected {len(candidates)} highlight candidate(s)")

    # Render
    compositor = VideoCompositor(profile_name=profile_name)
    processed_clips: list[Clip] = []

    for idx, cand in enumerate(candidates, 1):
        filename = f"clip_highlight_{idx}_of_{len(candidates)}_9x16.mp4" if len(candidates) > 1 else "test_output_9x16.mp4"
        out_path = storage.clips_dir / filename
        logger.info(f"[ClipPipeline] Rendering highlight {idx}/{len(candidates)} [{cand.start:.1f}s -> {cand.end:.1f}s]")
        rendered = await compositor.render_short(downloaded, cand.start, cand.end, out_path)

        clip = Clip(
            asset_id=asset.id,
            start_time=cand.start,
            end_time=cand.end,
            overall_score=cand.score,
            scores_breakdown=cand.breakdown,
            rendered_clip_path=rendered,
            status=ClipStatus.RENDERED,
        )
        clip.title = generate_title(asset, clip, profile_name=profile_name)
        clip.tags = generate_tags(category=profile_name)
        clip.description = generate_description(asset, clip.tags, profile_name=profile_name)

        await insert_clip(db, clip)
        processed_clips.append(clip)
        logger.success(f"[ClipPipeline] Successfully queued clip: '{clip.title}' -> {rendered.name}")

    await db.close()
    return processed_clips
