"""Robust media downloader using yt-dlp with retries and format selection."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
from pathlib import Path
from loguru import logger


def _get_ytdlp_bin() -> str:
    """Find the yt-dlp executable in the active virtualenv or system PATH."""
    venv_bin = Path(sys.executable).parent / "yt-dlp"
    if venv_bin.exists():
        return str(venv_bin)
    system_bin = shutil.which("yt-dlp")
    if system_bin:
        return system_bin
    return "yt-dlp"


class MediaDownloader:
    """Wrapper around yt-dlp for downloading clips and stream segments."""

    @staticmethod
    async def extract_info(url: str) -> dict:
        """Extract video metadata without downloading media."""
        ytdlp_bin = _get_ytdlp_bin()
        cmd = [
            ytdlp_bin,
            "--dump-single-json",
            "--no-playlist",
            "--no-warnings",
            url,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"yt-dlp metadata extraction failed: {stderr.decode()[:300]}")
        return json.loads(stdout.decode())

    @staticmethod
    async def download_media(
        url: str,
        output_path: Path,
        max_duration: int | None = None,
        timeout_seconds: int = 3600,
    ) -> Path:
        """
        Download video from any supported URL (Twitch, Kick, YouTube, etc.) to output_path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and output_path.stat().st_size > 1024:
            logger.debug(f"[Downloader] Cache hit: {output_path.name}")
            return output_path

        # Parallel multi-fragment download with aria2c acceleration for multi-hour HLS streams
        format_spec = "best[height<=720]/bestvideo[height<=720]+bestaudio/best[ext=mp4]/best"
        ytdlp_bin = _get_ytdlp_bin()
        cmd = [
            ytdlp_bin,
            "--no-playlist",
            "--no-warnings",
            "--concurrent-fragments", "10",
            "--retries", "10",
            "--fragment-retries", "10",
            "-f", format_spec,
            "--merge-output-format", "mp4",
        ]
        if shutil.which("aria2c"):
            cmd.extend(["--downloader", "aria2c", "--downloader-args", "aria2c:-x 16 -s 16 -k 1M -j 16"])
        cmd.extend(["-o", str(output_path), url])

        logger.info(f"[Downloader] Starting download: {url} -> {output_path.name}")
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            if proc.returncode != 0:
                err_msg = stderr.decode()[:400]
                raise RuntimeError(f"yt-dlp download failed with code {proc.returncode}: {err_msg}")

            if not output_path.exists():
                # Sometimes yt-dlp might append an extension or produce a slightly different filename
                candidates = list(output_path.parent.glob(f"{output_path.stem}.*"))
                if candidates:
                    return candidates[0]
                raise FileNotFoundError(f"Expected download output {output_path} not found")

            logger.info(f"[Downloader] Download completed: {output_path.name} ({output_path.stat().st_size / (1024**2):.1f} MB)")
            return output_path

        except asyncio.TimeoutError:
            if proc:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
            # Clean up partial download fragments
            for part in output_path.parent.glob(f"{output_path.stem}*.*part*"):
                try:
                    part.unlink()
                except Exception:
                    pass
            raise TimeoutError(f"Download timed out after {timeout_seconds}s for {url}")
        except Exception as exc:
            # Clean up partial download fragments on any failure
            for part in output_path.parent.glob(f"{output_path.stem}*.*part*"):
                try:
                    part.unlink()
                except Exception:
                    pass
            raise exc

    @staticmethod
    async def download_audio_only(
        url: str,
        output_path: Path,
        timeout_seconds: int = 1800,
    ) -> Path:
        """
        Download only the lightweight audio stream of a stream/video (ideal for multi-hour VODs).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and output_path.stat().st_size > 1024:
            logger.debug(f"[Downloader] Audio cache hit: {output_path.name}")
            return output_path

        ytdlp_bin = _get_ytdlp_bin()
        format_spec = "Audio_Only/bestaudio[ext=m4a]/bestaudio/best[ext=mp3]/best"
        cmd = [
            ytdlp_bin,
            "--no-playlist",
            "--no-warnings",
            "--concurrent-fragments", "5",
            "--retries", "10",
            "--fragment-retries", "10",
            "-f", format_spec,
            "-o", str(output_path),
            url,
        ]

        logger.info(f"[Downloader] Downloading audio track only: {url} -> {output_path.name}")
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            if proc.returncode != 0:
                err_msg = stderr.decode()[:400]
                raise RuntimeError(f"yt-dlp audio download failed with code {proc.returncode}: {err_msg}")

            if not output_path.exists():
                candidates = list(output_path.parent.glob(f"{output_path.stem}.*"))
                if candidates:
                    return candidates[0]
                raise FileNotFoundError(f"Expected audio download {output_path} not found")

            logger.info(f"[Downloader] Audio download completed: {output_path.name} ({output_path.stat().st_size / (1024**2):.1f} MB)")
            return output_path
        except asyncio.TimeoutError:
            if proc:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
            for part in output_path.parent.glob(f"{output_path.stem}*.*part*"):
                try:
                    part.unlink()
                except Exception:
                    pass
            raise TimeoutError(f"Audio download timed out after {timeout_seconds}s for {url}")
        except Exception as exc:
            for part in output_path.parent.glob(f"{output_path.stem}*.*part*"):
                try:
                    part.unlink()
                except Exception:
                    pass
            raise exc

    @staticmethod
    async def download_section(
        url: str,
        start_sec: float,
        end_sec: float,
        output_path: Path,
        timeout_seconds: int = 600,
    ) -> Path:
        """
        Download only a specific time slice (*start-end) of a long stream/VOD without downloading the whole video.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and output_path.stat().st_size > 1024:
            logger.debug(f"[Downloader] Section cache hit: {output_path.name}")
            return output_path

        start_int = int(max(0, round(start_sec)))
        end_int = int(round(end_sec))
        section_spec = f"*{start_int}-{end_int}"

        ytdlp_bin = _get_ytdlp_bin()
        format_spec = "1080p60/bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best"
        cmd = [
            ytdlp_bin,
            "--no-playlist",
            "--no-warnings",
            "--download-sections", section_spec,
            "--retries", "10",
            "--fragment-retries", "10",
            "-f", format_spec,
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            url,
        ]

        logger.info(f"[Downloader] Downloading section {section_spec}: {url} -> {output_path.name}")
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            if proc.returncode != 0:
                err_msg = stderr.decode()[:400]
                raise RuntimeError(f"yt-dlp section download failed with code {proc.returncode}: {err_msg}")

            if not output_path.exists():
                candidates = list(output_path.parent.glob(f"{output_path.stem}.*"))
                if candidates:
                    return candidates[0]
                raise FileNotFoundError(f"Expected section download {output_path} not found")

            logger.info(f"[Downloader] Section download completed: {output_path.name} ({output_path.stat().st_size / (1024**2):.1f} MB)")
            return output_path
        except asyncio.TimeoutError:
            if proc:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
            for part in output_path.parent.glob(f"{output_path.stem}*.*part*"):
                try:
                    part.unlink()
                except Exception:
                    pass
            raise TimeoutError(f"Section download timed out after {timeout_seconds}s for {url}")
        except Exception as exc:
            for part in output_path.parent.glob(f"{output_path.stem}*.*part*"):
                try:
                    part.unlink()
                except Exception:
                    pass
            raise exc
