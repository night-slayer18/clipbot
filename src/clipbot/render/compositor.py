"""Master video compositor — orchestrates cutting, 9:16 reframing, and burned karaoke subtitles."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from loguru import logger

from clipbot.core.config import get_config, load_profile
from clipbot.core.media import probe_media_resolution
from clipbot.detectors.facecam import FacecamDetector
from clipbot.render.captions import CaptionGenerator
from clipbot.render.cut import cut_clip
from clipbot.render.reframe import reframe_blur_canvas, reframe_split_stack


class VideoCompositor:
    """End-to-end renderer producing high-retention 9:16 vertical shorts."""

    def __init__(self, profile_name: str = "gta_chaos") -> None:
        self.profile = load_profile(profile_name)
        self.facecam_detector = FacecamDetector()
        self.caption_gen = CaptionGenerator(model_size="base")
        self.config = get_config()

    async def render_short(
        self,
        source_media: Path,
        start_time: float,
        end_time: float,
        output_path: Path,
        force_layout: Optional[str] = None,
    ) -> Path:
        """
        Produce a finished 1080x1920 vertical video with burned karaoke subtitles.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = output_path.parent / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        clip_id = output_path.stem
        raw_trimmed = temp_dir / f"{clip_id}_trimmed.mp4"
        reframed_video = temp_dir / f"{clip_id}_9x16.mp4"
        ass_path = temp_dir / f"{clip_id}_subs.ass"

        try:
            # 1. Trim the clip segment
            logger.info(f"[Compositor] Trimming {source_media.name} [{start_time:.1f}s -> {end_time:.1f}s]")
            await cut_clip(source_media, start_time, end_time, raw_trimmed, fast=False)

            # 2. Layout selection & Reframing
            layout = force_layout or self.profile.get("render", {}).get("layout", "split_stack")
            facecam = None
            if layout == "split_stack":
                facecam = await self.facecam_detector.detect(raw_trimmed)

            # Gate Split-Stack by confidence (minimum 0.60) to prevent false-positives
            if layout == "split_stack" and facecam is not None and facecam.confidence >= 0.60:
                src_w, src_h = probe_media_resolution(raw_trimmed)
                logger.info(
                    f"[Compositor] Using Split-Stack layout with facecam ({facecam.to_ffmpeg_crop(src_w, src_h)}) "
                    f"[confidence={facecam.confidence:.2f}, quadrant={facecam.quadrant}, src={src_w}x{src_h}]"
                )
                # Dynamically calculate gameplay crop matching exact 1080:1248 bottom slot ratio for any resolution
                bottom_aspect = 1080.0 / 1248.0
                crop_w = int(round(src_h * bottom_aspect))
                crop_w = (crop_w // 2) * 2
                crop_w = min(crop_w, src_w)
                crop_x = int(round((src_w - crop_w) / 2.0))
                crop_x = (crop_x // 2) * 2
                gameplay_crop = f"{crop_w}:{src_h}:{crop_x}:0"

                await reframe_split_stack(
                    source=raw_trimmed,
                    output=reframed_video,
                    top_crop=facecam.to_ffmpeg_crop(src_w, src_h),
                    bottom_crop=gameplay_crop,
                )
            else:
                if layout == "split_stack":
                    conf_val = getattr(facecam, "confidence", 0.0)
                    logger.info(
                        f"[Compositor] Corner facecam not locked (confidence={conf_val:.2f} < 0.60). "
                        "Falling back to Blur-Canvas layout (Safe Layout)."
                    )
                else:
                    logger.info(f"[Compositor] Using Blur-Canvas layout (layout={layout})")
                await reframe_blur_canvas(source=raw_trimmed, output=reframed_video)

            # 3. Generate word-level karaoke subtitles
            generated_ass = await self.caption_gen.generate_ass(raw_trimmed, ass_path)

            # 4. Burn subtitles if available
            if generated_ass and generated_ass.exists():
                logger.info(f"[Compositor] Burning karaoke subtitles into {output_path.name}")
                await self._burn_subtitles(reframed_video, generated_ass, output_path)
            else:
                # If no speech or whisper failed, use the clean reframed video
                logger.info("[Compositor] No subtitles to burn; saving clean reframed video")
                reframed_video.replace(output_path)

            logger.success(f"[Compositor] Render complete: {output_path.name} ({output_path.stat().st_size / (1024**2):.1f} MB)")
            return output_path

        finally:
            # Cleanup intermediate temp files
            for p in [raw_trimmed, reframed_video, ass_path]:
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass

    async def _burn_subtitles(self, video_path: Path, ass_path: Path, output_path: Path) -> None:
        """Burn ASS subtitles into the video using FFmpeg."""
        # Escape path for FFmpeg filter
        escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:").replace("'", "'\\\\\\''")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"ass='{escaped_ass}'",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-b:v", "8M", "-maxrate", "12M", "-bufsize", "16M",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(f"[Compositor] Subtitle burn failed, copying video without subs: {stderr.decode()[:300]}")
            video_path.replace(output_path)
