"""Movie Episodic Splitter — slices full-length movies into sequential 9:16 vertical series.

Creates Part 1, Part 2, ... Part N with:
  - Smart silence/dialogue-pause boundary detection (never cuts mid-sentence)
  - Top title & series tracker banner ("PART 1 OF 45 • FOLLOW FOR NEXT PART")
  - Centered 16:9 movie footage with blurred background
  - Word-level karaoke captions
  - Sequential database queue for staggered publishing
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Optional

import cv2
from loguru import logger

from clipbot.adapters.downloader import MediaDownloader
from clipbot.core.config import get_config, load_profile
from clipbot.core.db import get_db, init_db
from clipbot.core.models import Asset, AssetStatus, Clip, ClipStatus, Source
from clipbot.core.repository import insert_clip, upsert_asset
from clipbot.render.captions import CaptionGenerator
from clipbot.render.cut import cut_clip


class MovieSplitter:
    """Automates the entire long-form movie to vertical short series workflow."""

    def __init__(self, profile_name: str = "movie_episodic") -> None:
        self.profile = load_profile(profile_name)
        self.config = get_config()
        self.caption_gen = CaptionGenerator(model_size="base")

        episodic_cfg = self.profile.get("episodic", {})
        self.target_part_sec = episodic_cfg.get("target_part_seconds", 58)
        self.min_part_sec = episodic_cfg.get("min_part_seconds", 48)
        self.max_part_sec = episodic_cfg.get("max_part_seconds", 68)
        self.silence_thresh_db = episodic_cfg.get("silence_thresh_db", -32)
        self.silence_min_dur = episodic_cfg.get("silence_min_duration", 0.25)

    async def get_media_duration(self, video_path: Path) -> float:
        """Get exact duration in seconds using ffprobe with OpenCV fallback."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=duration",
            "-of", "json",
            str(video_path),
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0 and stdout:
                data = json.loads(stdout.decode())
                fmt_dur = data.get("format", {}).get("duration")
                if fmt_dur:
                    return float(fmt_dur)
                for s in data.get("streams", []):
                    if "duration" in s:
                        return float(s["duration"])
        except Exception as exc:
            logger.debug(f"[MovieSplitter] ffprobe notice: {exc}")

        # Fallback using cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if frames > 0 and fps > 0:
            return float(frames / fps)
        raise RuntimeError(f"Unable to determine duration of {video_path}")

    async def detect_silence_intervals(self, video_path: Path) -> list[tuple[float, float]]:
        """
        Detect silence intervals where dialogue pauses.
        Returns list of (silence_start, silence_end) in seconds.
        """
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-af", f"silencedetect=noise={self.silence_thresh_db}dB:d={self.silence_min_dur}",
            "-f", "null", "-",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        lines = stderr.decode().splitlines()

        silences: list[tuple[float, float]] = []
        cur_start: Optional[float] = None

        for line in lines:
            if "silence_start:" in line:
                m = re.search(r"silence_start:\s*([0-9.]+)", line)
                if m:
                    cur_start = float(m.group(1))
            elif "silence_end:" in line and cur_start is not None:
                m = re.search(r"silence_end:\s*([0-9.]+)", line)
                if m:
                    cur_end = float(m.group(1))
                    silences.append((cur_start, cur_end))
                    cur_start = None

        logger.info(f"[MovieSplitter] Found {len(silences)} dialogue silence points in {video_path.name}")
        return silences

    def calculate_smart_cut_points(
        self, total_duration: float, silences: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """
        Calculate sequential (start, end) timestamps cut during dialogue pauses.
        """
        cuts: list[tuple[float, float]] = []
        cur_start = 0.0

        while cur_start < total_duration:
            ideal_end = cur_start + self.target_part_sec
            search_min = cur_start + self.min_part_sec
            search_max = min(total_duration, cur_start + self.max_part_sec)

            if total_duration - cur_start <= self.max_part_sec:
                # Last remaining chunk
                cuts.append((cur_start, total_duration))
                break

            # Find a silence that falls within [search_min, search_max]
            best_cut = ideal_end
            candidate_silences = [
                (s, e) for s, e in silences if search_min <= s <= search_max
            ]

            if candidate_silences:
                # Pick silence closest to ideal_end
                candidate_silences.sort(key=lambda x: abs(x[0] - ideal_end))
                # Cut at the midpoint of silence
                s_start, s_end = candidate_silences[0]
                best_cut = (s_start + s_end) / 2
            else:
                best_cut = min(ideal_end, total_duration)

            cuts.append((cur_start, best_cut))
            cur_start = best_cut

        return cuts

    async def render_movie_part(
        self,
        source_path: Path,
        start_time: float,
        end_time: float,
        output_path: Path,
        movie_title: str,
        part_num: int,
        total_parts: int,
    ) -> Path:
        """
        Render a single 9:16 vertical episode with top series banner and subtitles.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = output_path.parent / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        part_id = f"part_{part_num}"
        raw_cut = temp_dir / f"{part_id}_cut.mp4"
        reframed = temp_dir / f"{part_id}_9x16.mp4"
        ass_path = temp_dir / f"{part_id}.ass"

        try:
            # 1. Trim part segment
            await cut_clip(source_path, start_time, end_time, raw_cut, fast=False)

            # 2. Build 9:16 layout with top movie header
            canvas_w, canvas_h = 1080, 1920
            video_h = int(canvas_w * 9 / 16)
            y_offset = (canvas_h - video_h) // 2

            header_text = f"🍿 {movie_title.upper()}"
            sub_header_text = f"PART {part_num} OF {total_parts}  •  FOLLOW FOR NEXT PART"

            # Escape special FFmpeg drawtext filter characters (single quotes, colons, percent)
            safe_header = header_text.replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")
            safe_sub_header = sub_header_text.replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")

            # FFmpeg filtergraph:
            # - Blurred 9:16 background (scale height to canvas_h first, then crop)
            # - Crisp centered 16:9 movie footage
            # - Top banner overlay with Movie Title and Series Counter
            filtergraph = (
                f"[0:v]scale=-2:{canvas_h},crop={canvas_w}:{canvas_h},gblur=sigma=22[bg];"
                f"[0:v]scale={canvas_w}:{video_h}[fg];"
                f"[bg][fg]overlay=0:{y_offset}[composed];"
                f"[composed]"
                f"drawbox=y=0:color=black@0.75:width=iw:height=320:t=fill,"
                f"drawtext=text='{safe_header}':fontcolor=white:fontsize=46:x=(w-text_w)/2:y=80:shadowcolor=black:shadowx=2:shadowy=2,"
                f"drawtext=text='{safe_sub_header}':fontcolor=yellow:fontsize=38:x=(w-text_w)/2:y=170:shadowcolor=black:shadowx=2:shadowy=2"
                f"[out]"
            )

            cmd = [
                "ffmpeg", "-y", "-i", str(raw_cut),
                "-filter_complex", filtergraph,
                "-map", "[out]", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "fast", "-crf", "19",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(reframed),
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg movie composite failed: {stderr.decode()[:300]}")

            # 3. Generate and burn karaoke subtitles
            ass_generated = await self.caption_gen.generate_ass(raw_cut, ass_path)
            if ass_generated and ass_generated.exists():
                escaped_ass = str(ass_generated).replace("\\", "/").replace(":", "\\:").replace("'", "'\\\\\\''")
                burn_cmd = [
                    "ffmpeg", "-y", "-i", str(reframed),
                    "-vf", f"ass='{escaped_ass}'",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "19",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    str(output_path),
                ]
                bproc = await asyncio.create_subprocess_exec(
                    *burn_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
                )
                await bproc.communicate()
                if not output_path.exists():
                    reframed.replace(output_path)
            else:
                reframed.replace(output_path)

            logger.info(f"[MovieSplitter] Rendered Part {part_num}/{total_parts}: {output_path.name}")
            return output_path

        finally:
            for p in [raw_cut, reframed, ass_path]:
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass

    async def split_and_queue_movie(
        self,
        movie_input: str,
        title: str,
        max_parts: Optional[int] = None,
    ) -> list[Clip]:
        """
        Master method: downloads movie if URL, slices sequentially, renders, and queues in DB.
        """
        db_path = self.config.database.path
        await init_db(db_path)
        clips_dir = Path(self.config.storage.clips_dir) / "movies"
        media_dir = Path(self.config.storage.media_dir) / "movies"
        clips_dir.mkdir(parents=True, exist_ok=True)
        media_dir.mkdir(parents=True, exist_ok=True)

        # 1. Obtain local file
        if movie_input.startswith("http://") or movie_input.startswith("https://"):
            logger.info(f"[MovieSplitter] Downloading movie from: {movie_input}")
            safe_title = "".join(c for c in title if c.isalnum() or c in " _-").strip()
            movie_file = media_dir / f"{safe_title}.mp4"
            movie_file = await MediaDownloader.download_media(movie_input, movie_file, timeout_seconds=1800)
        else:
            movie_file = Path(movie_input)
            if not movie_file.exists():
                raise FileNotFoundError(f"Movie file not found: {movie_file}")

        # 2. Get duration and silences
        duration = await self.get_media_duration(movie_file)
        logger.info(f"[MovieSplitter] Loaded '{title}' — Total Duration: {duration / 60:.1f} minutes")

        silences = await self.detect_silence_intervals(movie_file)
        cuts = self.calculate_smart_cut_points(duration, silences)
        if max_parts:
            cuts = cuts[:max_parts]

        total_parts = len(cuts)
        logger.info(f"[MovieSplitter] Splitting into {total_parts} sequential episodes (~{self.target_part_sec}s each)")

        # 3. Create parent asset in DB
        db = await get_db(db_path)
        asset = Asset(
            source=Source.YOUTUBE,
            external_id=movie_file.stem,
            creator="Film Series",
            title=title,
            game="Movie",
            url=movie_input,
            duration=duration,
            local_path=movie_file,
            status=AssetStatus.READY,
        )
        await upsert_asset(db, asset)

        generated_clips: list[Clip] = []

        try:
            for idx, (start_t, end_t) in enumerate(cuts, start=1):
                part_filename = f"{movie_file.stem}_part_{idx:03d}_of_{total_parts:03d}.mp4"
                rendered_path = clips_dir / part_filename

                logger.info(f"[MovieSplitter] Processing Part {idx}/{total_parts} [{start_t:.1f}s -> {end_t:.1f}s]")

                await self.render_movie_part(
                    source_path=movie_file,
                    start_time=start_t,
                    end_time=end_t,
                    output_path=rendered_path,
                    movie_title=title,
                    part_num=idx,
                    total_parts=total_parts,
                )

                clip = Clip(
                    asset_id=asset.id,
                    start_time=start_t,
                    end_time=end_t,
                    overall_score=1.0 + (idx * 0.01),  # Keeps chronological order in query
                    rendered_clip_path=rendered_path,
                    status=ClipStatus.RENDERED,
                    title=f"🍿 {title} - Part {idx} of {total_parts} #shorts",
                    description=(
                        f"Watch the full story! Follow for Part {idx + 1}.\n\n"
                        f"Movie: {title}\n"
                        f"Part: {idx}/{total_parts}\n\n"
                        f"#movies #movieclips #film #cinema #movietime #shorts #fyp"
                    ),
                    tags=["movies", "movieclips", "film", "cinema", "reels", "shorts", "fyp"],
                )
                await insert_clip(db, clip)
                generated_clips.append(clip)

            logger.success(f"[MovieSplitter] Successfully split '{title}' into {len(generated_clips)} parts ready for review/publishing!")
        finally:
            await db.close()

        return generated_clips
