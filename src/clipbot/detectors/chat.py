"""Chat velocity detector — bins chat messages into 1Hz signal with emote weighting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

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
        if asset.chat_path is None or not asset.chat_path.exists():
            logger.debug(f"[Chat] No chat log for {asset.external_id}")
            duration = int(asset.duration or 60)
            return {"chat_velocity": np.zeros(duration, dtype=np.float32)}

        # Chat log format: each line is a JSON object with 'offset' (seconds) and 'message'
        duration = int(asset.duration or 3600)
        velocity = np.zeros(duration, dtype=np.float32)

        with asset.chat_path.open() as f:
            for line in f:
                try:
                    msg = json.loads(line.strip())
                    offset = int(msg.get("offset", msg.get("time_in_seconds", 0)))
                    if 0 <= offset < duration:
                        text = msg.get("message", msg.get("body", ""))
                        weight = self._message_weight(text)
                        velocity[offset] += weight
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        logger.debug(f"[Chat] {asset.external_id}: max velocity={velocity.max():.1f} msgs/sec")
        return {"chat_velocity": velocity}

    def _message_weight(self, text: str) -> float:
        words = text.split()
        total = sum(EMOTE_WEIGHTS.get(w, DEFAULT_WEIGHT) for w in words)
        return total / max(len(words), 1)
