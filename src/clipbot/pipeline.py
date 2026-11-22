"""Master pipeline orchestrator — end-to-end discover -> download -> score -> render cycle."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import aiosqlite
import numpy as np
from loguru import logger

from clipbot.adapters.kick import KickAdapter
from clipbot.adapters.twitch import TwitchAdapter
from clipbot.adapters.youtube import YouTubeAdapter
from clipbot.core.config import get_config, load_profile
from clipbot.core.db import get_db, init_db
from clipbot.core.media import probe_media_duration
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


async def run_cycle(profile_name: str = "gta6_chaos", auto_render: bool = True) -> list[Clip]:
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

    profile = load_profile(profile_name)

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
    transcript_det = TranscriptDetector(model_size="base", profile_name=profile_name)
    compositor = VideoCompositor(profile_name=profile_name)

    db = await get_db(db_path)
    processed_clips: list[Clip] = []

    try:
        # 2. Process assets up to cycle limit
        limit = config.pipeline.max_clips_per_cycle
        for asset in all_discovered[:limit]:
            # Fix Flaw 8: Skip if asset was already fully processed in a previous cycle
            # Check SQLite directly for existing clips or completed status regardless of local cache pruning
            existing = await get_asset_by_external(db, asset.source.value, asset.external_id)
            if existing:
                async with db.execute("SELECT COUNT(*) FROM clips WHERE asset_id = ?", (existing.id,)) as cur:
                    clip_count = (await cur.fetchone())[0]
                if clip_count > 0 or existing.status in (AssetStatus.SCORED, AssetStatus.ARCHIVED):
                    logger.info(f"[Pipeline] Skipping already processed asset ({clip_count} clips exist): {asset.title[:50]}")
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
                # Accurately probe media duration if missing
                if not asset.duration or asset.duration <= 0:
                    asset.duration = probe_media_duration(asset.local_path)
                await update_asset_status(db, asset.id, AssetStatus.READY)
            except Exception as exc:
                logger.error(f"[Pipeline] Download failed for {asset.external_id}: {exc}")
                await update_asset_status(db, asset.id, AssetStatus.ERROR)
                continue

            # Fix Flaw 6: Wire chat fetch
            try:
                chat_file = await adapter.fetch_chat(asset, storage.media_dir)
                if chat_file and chat_file.exists():
                    asset.chat_path = chat_file
            except Exception as exc:
                logger.debug(f"[Pipeline] Chat fetch notice: {exc}")

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

                candidate_windows = None
                if asset.duration and asset.duration > 300 and "audio_rms" in signals:
                    from clipbot.scoring.combine import _find_local_peaks, _zscore
                    coarse_len = min(len(signals["audio_rms"]), len(signals.get("visual_impact", signals["audio_rms"])))
                    rms_c = _zscore(signals["audio_rms"][:coarse_len])
                    impact_c = _zscore(signals.get("visual_impact", np.zeros(coarse_len, dtype=np.float32))[:coarse_len])
                    motion_c = _zscore(signals.get("visual_motion", np.zeros(coarse_len, dtype=np.float32))[:coarse_len])
                    combined_coarse = rms_c + impact_c + motion_c
                    coarse_peaks = _find_local_peaks(combined_coarse, min_distance=max(20, int(config.pipeline.min_gap_seconds * 0.5)), min_score=0.3)
                    candidate_windows = [
                        (max(0.0, float(p - 15)), min(float(asset.duration), float(p + 35)))
                        for p in coarse_peaks[:profile.get("scoring", {}).get("top_n", 3) * 4]
                    ]

                try:
                    trans_sig = await transcript_det.signals(
                        asset, asset.local_path, profile_name=profile_name, candidate_windows=candidate_windows
                    )
                    signals.update(trans_sig)
                except Exception as exc:
                    logger.warning(f"[Pipeline] Transcript detector error: {exc}")

                # Cache signals in DB
                for name, arr in signals.items():
                    await upsert_signal(db, asset.id, name, arr)

            # Score & window candidate highlights
            scoring_top_n = profile.get("scoring", {}).get("top_n", 3)
            candidates = score_and_window(
                signals=signals,
                profile_name=profile_name,
                min_duration=config.pipeline.min_clip_duration,
                max_duration=config.pipeline.max_clip_duration,
                min_gap=-1.0,
                top_n=scoring_top_n,
                media_duration=asset.duration,
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

            await update_asset_status(db, asset.id, AssetStatus.SCORED)

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
    layout: Optional[str] = None,
) -> list[Clip]:
    """Execute the full highlights pipeline on a single URL, rendering up to max_clips."""
    config = get_config()
    db_path = config.database.path
    await init_db(db_path)

    storage = StorageManager(
        media_dir=config.storage.media_dir,
        clips_dir=config.storage.clips_dir,
        max_cache_gb=config.storage.max_cache_gb,
    )
    logger.info(f"[ClipPipeline] Processing highlights on {url} (max {max_clips} clips, layout={layout or 'auto'})")

    from clipbot.adapters.downloader import MediaDownloader
    import hashlib
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:10]
    temp_media = storage.media_dir / f"stream_{url_hash}.mp4"
    downloaded = await MediaDownloader.download_media(url, temp_media)

    media_dur = probe_media_duration(downloaded)
    asset = Asset(
        external_id=f"stream_{url_hash}",
        creator="Stream Creator",
        title=f"Stream Highlights {url_hash}",
        url=url,
        duration=media_dur,
        local_path=downloaded,
        status=AssetStatus.READY,
    )

    db = await get_db(db_path)
    await upsert_asset(db, asset)

    # Extract signals
    audio_det = AudioDetector()
    visual_det = VisualDetector()
    transcript_det = TranscriptDetector(model_size="base", profile_name=profile_name)

    logger.info("[ClipPipeline] Running multi-modal detectors...")
    signals = {}
    existing_signals = await get_signals(db, asset.id)
    if existing_signals:
        signals.update(existing_signals)
        logger.info(f"[ClipPipeline] Loaded {len(signals)} cached signal(s) from database for {asset.external_id}")

    if "audio_rms" not in signals:
        try:
            audio_sig = await audio_det.signals(asset, downloaded)
            signals.update(audio_sig)
            for name, arr in audio_sig.items():
                await upsert_signal(db, asset.id, name, arr)
        except Exception as exc:
            logger.warning(f"[ClipPipeline] Audio detector error: {exc}")

    if "visual_scene" not in signals:
        try:
            vis_sig = await visual_det.signals(asset, downloaded)
            signals.update(vis_sig)
            for name, arr in vis_sig.items():
                await upsert_signal(db, asset.id, name, arr)
        except Exception as exc:
            logger.warning(f"[ClipPipeline] Visual detector error: {exc}")

    candidate_windows = None
    if media_dur and media_dur > 300 and "audio_rms" in signals:
        from clipbot.scoring.combine import _find_local_peaks, _zscore
        coarse_len = min(len(signals["audio_rms"]), len(signals.get("visual_impact", signals["audio_rms"])))
        rms_c = _zscore(signals["audio_rms"][:coarse_len])
        impact_c = _zscore(signals.get("visual_impact", np.zeros(coarse_len, dtype=np.float32))[:coarse_len])
        motion_c = _zscore(signals.get("visual_motion", np.zeros(coarse_len, dtype=np.float32))[:coarse_len])
        combined_coarse = rms_c + impact_c + motion_c
        coarse_peaks = _find_local_peaks(combined_coarse, min_distance=max(20, int(config.pipeline.min_gap_seconds * 0.5)), min_score=0.3)
        candidate_windows = [
            (max(0.0, float(p - 15)), min(float(media_dur), float(p + 35)))
            for p in coarse_peaks[:max_clips * 4]
        ]

    if "transcript_hype" not in signals:
        try:
            trans_sig = await transcript_det.signals(asset, downloaded, profile_name=profile_name, candidate_windows=candidate_windows)
            signals.update(trans_sig)
            for name, arr in trans_sig.items():
                await upsert_signal(db, asset.id, name, arr)
        except Exception as exc:
            logger.warning(f"[ClipPipeline] Transcript detector error: {exc}")

    # Window candidates (ignoring intermission and credits)
    candidates = score_and_window(
        signals=signals,
        profile_name=profile_name,
        min_duration=config.pipeline.min_clip_duration,
        max_duration=config.pipeline.max_clip_duration,
        min_gap=-1.0,
        top_n=max_clips,
        media_duration=media_dur,
    )
    if not candidates:
        logger.warning("[ClipPipeline] No peaks found; using opening segment")
        sig_len = float(media_dur or 30.0)
        from clipbot.scoring.combine import ClipCandidate
        candidates = [ClipCandidate(start=0.0, end=min(30.0, sig_len), score=1.0, breakdown={})]

    logger.info(f"[ClipPipeline] Selected {len(candidates)} highlight candidate(s)")

    # Render
    compositor = VideoCompositor(profile_name=profile_name)
    processed_clips: list[Clip] = []

    for idx, cand in enumerate(candidates, 1):
        filename = f"clip_{url_hash}_{idx}_9x16.mp4"
        out_path = storage.clips_dir / filename
        logger.info(f"[ClipPipeline] Rendering highlight {idx}/{len(candidates)} [{cand.start:.1f}s -> {cand.end:.1f}s]")
        rendered = await compositor.render_short(downloaded, cand.start, cand.end, out_path, force_layout=layout)

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

    await update_asset_status(db, asset.id, AssetStatus.SCORED)
    await db.close()
    return processed_clips

