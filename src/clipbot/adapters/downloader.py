"""Robust media downloader using yt-dlp with retries and format selection."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from loguru import logger


class MediaDownloader:
    """Wrapper around yt-dlp for downloading clips and stream segments."""

    @staticmethod
    async def extract_info(url: str) -> dict:
        """Extract video metadata without downloading media."""
        cmd = [
            "yt-dlp",
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
        timeout_seconds: int = 300,
    ) -> Path:
        """
        Download video from any supported URL (Twitch, Kick, YouTube, etc.) to output_path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and output_path.stat().st_size > 1024:
            logger.debug(f"[Downloader] Cache hit: {output_path.name}")
            return output_path

        # Standard best quality mp4 format selector
        format_spec = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best"
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--no-warnings",
            "-f", format_spec,
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            url,
        ]

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
