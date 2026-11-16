"""Rate limiter and daily quota tracker for social media publishing across multiple accounts."""

from __future__ import annotations

from datetime import datetime, timezone
import aiosqlite
from loguru import logger

from clipbot.core.auth import load_accounts_config
from clipbot.core.config import get_config


class PublishRateLimiter:
    """Enforces daily upload limits and optimal posting hour windows per account."""

    def __init__(self, db_path: str = "data/state.db") -> None:
        self.db_path = db_path
        self.config = get_config()

    async def can_publish(self, platform: str, account_id: str = "default") -> tuple[bool, str]:
        """
        Check if we are allowed to post to `platform` on `account_id` right now.
        Returns: (can_post, reason)
        """
        accounts_cfg = load_accounts_config().get("accounts", {})
        plat_accounts = accounts_cfg.get(platform, [])

        # Default limit
        daily_limit = self.config.publish.posts_per_day.get(platform, 3)

        # Check account-specific limit if defined in accounts.yaml
        for acc in plat_accounts:
            if acc.get("id") == account_id and "daily_limit" in acc:
                daily_limit = acc["daily_limit"]
                break

        # Count posts made today UTC for this specific account
        today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT COUNT(*) FROM posts
                WHERE platform=? AND account_id=? AND status='published' AND published_at LIKE ?
                """,
                (platform, account_id, f"{today_prefix}%"),
            ) as cur:
                row = await cur.fetchone()
                published_today = row[0] if row else 0

        if published_today >= daily_limit:
            return False, f"Daily limit reached for {platform} [{account_id}] ({published_today}/{daily_limit})"

        return True, f"OK ({published_today}/{daily_limit} published today)"

    def is_preferred_hour(self) -> bool:
        """Check if current local hour is in the preferred spread hours."""
        current_hour = datetime.now().hour
        spread = self.config.publish.spread_hours
        return current_hour in spread
