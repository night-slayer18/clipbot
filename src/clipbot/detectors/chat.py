"""Chat velocity detector — bins chat messages into 1Hz signal with emote weighting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

from clipbot.core.media import probe_media_duration
from clipbot.core.models import Asset

# GTA-specific emote weight map
EMOTE_WEIGHTS: dict[str, float] = {
    # Hard laughs
    "KEKW": 1.5, "OMEGALUL": 1.5, "LUL": 1.3, "ICANT": 1.5,
    "💀": 1.5, "🤣": 1.4, "LMAO": 1.3,
    # Shock / drama
    "WTF": 1.3, "HUH": 1.3, "OMG": 1.2, "NOOO": 1.3,
    "monkaW": 1.3, "monkaS": 1.3, "W": 1.2, "L": 1.2, "CLIP": 1.4,
    # RP specific
    "bwo": 1.4, "CAUGHT": 1.4, "DESKCHAN": 1.3,
    # Hype
    "Pog": 1.1, "PogChamp": 1.1, "POGGERS": 1.1,
}

DEFAULT_WEIGHT = 1.0


class ChatDetector:
    name = "chat"

    async def signals(self, asset: Asset, media_path: Path) -> dict[str, np.ndarray]:
        probed_dur = probe_media_duration(media_path) if media_path and media_path.exists() else 0.0
        duration = int(max(probed_dur or 0, asset.duration or 0, 10))

        if asset.chat_path is None or not asset.chat_path.exists():
            logger.debug(f"[Chat] No chat log for {asset.external_id}")
            return {"chat_velocity": np.zeros(duration, dtype=np.float32)}

        velocity = np.zeros(duration, dtype=np.float32)

        # Support JSONL and standard yt-dlp comments JSON structure
        try:
            with asset.chat_path.open() as f:
                first_char = f.read(1)
                f.seek(0)
                if first_char == "{":
                    # Check if entire file is a single JSON object (yt-dlp comments format)
                    try:
                        data = json.load(f)
                        comments = data.get("comments", [])
                        if comments:
                            for c in comments:
                                offset = int(c.get("time_in_seconds", c.get("time", 0)))
                                if 0 <= offset < duration:
                                    text = c.get("text", c.get("body", ""))
                                    velocity[offset] += self._message_weight(text)
                            logger.debug(f"[Chat] {asset.external_id}: parsed {len(comments)} comments from JSON")
                            return {"chat_velocity": velocity}
                    except json.JSONDecodeError:
                        f.seek(0)

                # Line-by-line JSONL format
                f.seek(0)
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        msg = json.loads(line_str)
                        offset = int(msg.get("offset", msg.get("time_in_seconds", msg.get("time", 0))))
                        if 0 <= offset < duration:
                            text = msg.get("message", msg.get("body", msg.get("text", "")))
                            weight = self._message_weight(text)
                            velocity[offset] += weight
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception as exc:
            logger.warning(f"[Chat] Error reading chat log {asset.chat_path.name}: {exc}")

        logger.debug(f"[Chat] {asset.external_id}: max velocity={velocity.max():.1f} msgs/sec")
        return {"chat_velocity": velocity}

    def _message_weight(self, text: str) -> float:
        words = text.split()
        total = sum(EMOTE_WEIGHTS.get(w, DEFAULT_WEIGHT) for w in words)
        return total / max(len(words), 1)
