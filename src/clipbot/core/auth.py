"""OAuth 2.0 multi-account authentication manager for YouTube, Twitch, Instagram, TikTok, and Kick."""

from __future__ import annotations

import asyncio
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import yaml
from loguru import logger

from clipbot.core.config import CONFIG_DIR, get_settings


CREDENTIALS_DIR = Path("data/credentials")
CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)


def get_token_path(platform: str, account: str = "default") -> Path:
    """Return token JSON path for a specific platform and account ID."""
    if account == "default" or not account:
        return CREDENTIALS_DIR / f"{platform}_token.json"
    return CREDENTIALS_DIR / f"{platform}_{account}.json"


def load_accounts_config() -> dict[str, Any]:
    acc_file = CONFIG_DIR / "accounts.yaml"
    if acc_file.exists():
        with acc_file.open() as f:
            return yaml.safe_load(f) or {}
    return {}


class CallbackServer:
    """Lightweight local HTTP server to receive OAuth redirect codes."""

    def __init__(self, port: int = 8765) -> None:
        self.port = port
        self.code: Optional[str] = None
        self.error: Optional[str] = None

    def wait_for_code(self, timeout: int = 120) -> str:
        server_instance = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)

                if "code" in params:
                    server_instance.code = params["code"][0]
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"""
                        <html><body style="font-family:sans-serif;background:#0b0d10;color:#ffe600;text-align:center;padding:50px;">
                        <h2>Authentication Successful!</h2>
                        <p style="color:#eee;">Account connected. You can close this tab and return to the ClipBot terminal.</p>
                        </body></html>
                    """)
                else:
                    server_instance.error = params.get("error", ["Unknown error"])[0]
                    self.send_response(400)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"Authentication Failed.")

            def log_message(self, format, *args):
                pass

        class ReusableHTTPServer(HTTPServer):
            allow_reuse_address = True

        httpd = ReusableHTTPServer(("127.0.0.1", self.port), Handler)
        httpd.timeout = timeout
        logger.info(f"[OAuth] Waiting for browser callback on http://127.0.0.1:{self.port} (timeout {timeout}s)...")
        httpd.handle_request()
        httpd.server_close()

        if self.code:
            return self.code
        raise TimeoutError(f"OAuth callback timed out or failed: {self.error}")


# ── YouTube OAuth (Multi-Account) ─────────────────────────────────────────────

class YouTubeAuth:
    SCOPES = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube.readonly",
    ]

    @classmethod
    def authenticate(cls, secrets_file: str | Path | None = None, account: str = "default") -> bool:
        """Run interactive Google OAuth flow for a specific YouTube account."""
        from google_auth_oauthlib.flow import InstalledAppFlow

        settings = get_settings()
        secrets_candidate = secrets_file or settings.youtube_oauth_client_secrets_file
        if not secrets_candidate or not str(secrets_candidate).strip():
            logger.error(
                "[YouTube OAuth] No client secrets file specified.\n"
                "Provide one via --secrets path/to/client_secrets.json or set CLIPBOT_YOUTUBE_OAUTH_CLIENT_SECRETS_FILE in .env"
            )
            return False

        file_to_use = Path(secrets_candidate)
        if not file_to_use.is_file():
            logger.error(
                f"[YouTube OAuth] Client secrets file not found or is not a file: {file_to_use}\n"
                "Download your OAuth 2.0 Client ID JSON from Google Cloud Console."
            )
            return False

        flow = InstalledAppFlow.from_client_secrets_file(str(file_to_use), scopes=cls.SCOPES)
        logger.info(f"[YouTube OAuth] Launching browser to authenticate account: '{account}'...")
        creds = flow.run_local_server(port=8088, prompt="consent", access_type="offline")

        token_file = get_token_path("youtube", account)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        with token_file.open("w") as f:
            f.write(creds.to_json())

        logger.success(f"[YouTube OAuth] Saved token to {token_file}")
        return cls.verify(account)

    @classmethod
    def verify(cls, account: str = "default") -> bool:
        token_file = get_token_path("youtube", account)
        if not token_file.exists():
            return False
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            creds = Credentials.from_authorized_user_file(str(token_file), cls.SCOPES)
            youtube = build("youtube", "v3", credentials=creds)
            res = youtube.channels().list(part="snippet", mine=True).execute()
            items = res.get("items", [])
            if items:
                title = items[0]["snippet"]["title"]
                logger.success(f"[YouTube OAuth] [{account}] Connected Channel: '{title}'")
                return True
        except Exception as exc:
            logger.warning(f"[YouTube OAuth] [{account}] Verification error: {exc}")
        return False


# ── Twitch OAuth (Multi-Account) ──────────────────────────────────────────────

class TwitchAuth:
    REDIRECT_URI = "http://127.0.0.1:8765/callback/twitch"
    SCOPES = ["clips:edit", "user:read:email"]

    @classmethod
    async def authenticate(cls, account: str = "default") -> bool:
        """Interactive Twitch user authorization code exchange."""
        settings = get_settings()
        client_id = settings.twitch_client_id
        client_secret = settings.twitch_client_secret

        if not client_id or not client_secret:
            logger.error("[Twitch OAuth] CLIPBOT_TWITCH_CLIENT_ID or CLIPBOT_TWITCH_CLIENT_SECRET missing in .env")
            return False

        params = {
            "client_id": client_id,
            "redirect_uri": cls.REDIRECT_URI,
            "response_type": "code",
            "scope": " ".join(cls.SCOPES),
        }
        auth_url = f"https://id.twitch.tv/oauth2/authorize?{urlencode(params)}"
        logger.info(f"[Twitch OAuth] Opening browser to authenticate account '{account}': {auth_url}")
        webbrowser.open(auth_url)

        server = CallbackServer(port=8765)
        code = await asyncio.get_event_loop().run_in_executor(None, server.wait_for_code)

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://id.twitch.tv/oauth2/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": cls.REDIRECT_URI,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        token_file = get_token_path("twitch", account)
        with token_file.open("w") as f:
            json.dump(data, f, indent=2)

        logger.success(f"[Twitch OAuth] Saved token to {token_file}")
        return await cls.verify(account)

    @classmethod
    async def verify(cls, account: str = "default") -> bool:
        token_file = get_token_path("twitch", account)
        if not token_file.exists():
            return False
        try:
            settings = get_settings()
            with token_file.open() as f:
                token_data = json.load(f)
            token = token_data.get("access_token")

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.twitch.tv/helix/users",
                    headers={
                        "Client-Id": settings.twitch_client_id,
                        "Authorization": f"Bearer {token}",
                    },
                )
                if resp.status_code == 200:
                    user = resp.json().get("data", [{}])[0]
                    logger.success(f"[Twitch OAuth] [{account}] Connected User: '{user.get('display_name')}'")
                    return True
        except Exception as exc:
            logger.warning(f"[Twitch OAuth] [{account}] Verification failed: {exc}")
        return False


# ── Instagram OAuth (Multi-Account) ───────────────────────────────────────────

class InstagramAuth:
    REDIRECT_URI = "http://127.0.0.1:8765/callback/instagram"

    @classmethod
    async def authenticate(cls, app_id: str, app_secret: str, account: str = "default") -> bool:
        """Interactive Meta / Instagram Graph API OAuth flow for a specific account."""
        scopes = "instagram_basic,instagram_content_publish,pages_show_list,pages_read_engagement"
        params = {
            "client_id": app_id,
            "redirect_uri": cls.REDIRECT_URI,
            "scope": scopes,
            "response_type": "code",
        }
        auth_url = f"https://www.facebook.com/v19.0/dialog/oauth?{urlencode(params)}"
        logger.info(f"[Instagram OAuth] Opening browser to authenticate account '{account}': {auth_url}")
        webbrowser.open(auth_url)

        server = CallbackServer(port=8765)
        code = await asyncio.get_event_loop().run_in_executor(None, server.wait_for_code)

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://graph.facebook.com/v19.0/oauth/access_token",
                params={
                    "client_id": app_id,
                    "client_secret": app_secret,
                    "redirect_uri": cls.REDIRECT_URI,
                    "code": code,
                },
            )
            resp.raise_for_status()
            short_token = resp.json().get("access_token")

            long_resp = await client.get(
                "https://graph.facebook.com/v19.0/oauth/access_token",
                params={
                    "grant_type": "fb_exchange_token",
                    "client_id": app_id,
                    "client_secret": app_secret,
                    "fb_exchange_token": short_token,
                },
            )
            long_resp.raise_for_status()
            data = long_resp.json()

        token_file = get_token_path("instagram", account)
        with token_file.open("w") as f:
            json.dump(data, f, indent=2)

        logger.success(f"[Instagram OAuth] Saved long-lived token to {token_file}")
        return await cls.verify(account)

    @classmethod
    async def verify(cls, account: str = "default") -> bool:
        token_file = get_token_path("instagram", account)
        if not token_file.exists():
            return False
        try:
            with token_file.open() as f:
                token_data = json.load(f)
            token = token_data.get("access_token")

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://graph.facebook.com/v19.0/me",
                    params={"access_token": token, "fields": "id,name"},
                )
                if resp.status_code == 200:
                    d = resp.json()
                    logger.success(f"[Instagram OAuth] [{account}] Connected Page: '{d.get('name')}'")
                    return True
        except Exception as exc:
            logger.warning(f"[Instagram OAuth] [{account}] Verification failed: {exc}")
        return False


# ── TikTok OAuth (Multi-Account) ──────────────────────────────────────────────

class TikTokAuth:
    REDIRECT_URI = "http://127.0.0.1:8765/callback/tiktok"

    @classmethod
    async def authenticate(cls, account: str = "default") -> bool:
        """Interactive TikTok Content Posting API authorization for a specific account."""
        settings = get_settings()
        client_key = settings.tiktok_client_key
        client_secret = settings.tiktok_client_secret

        if not client_key or not client_secret:
            logger.error("[TikTok OAuth] CLIPBOT_TIKTOK_CLIENT_KEY or CLIPBOT_TIKTOK_CLIENT_SECRET missing in .env")
            return False

        params = {
            "client_key": client_key,
            "scope": "video.upload,video.publish,user.info.basic",
            "response_type": "code",
            "redirect_uri": cls.REDIRECT_URI,
            "state": f"clipbot_{account}",
        }
        auth_url = f"https://www.tiktok.com/v2/auth/authorize/?{urlencode(params)}"
        logger.info(f"[TikTok OAuth] Opening browser for account '{account}': {auth_url}")
        webbrowser.open(auth_url)

        server = CallbackServer(port=8765)
        code = await asyncio.get_event_loop().run_in_executor(None, server.wait_for_code)

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://open.tiktokapis.com/v2/oauth/token/",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "client_key": client_key,
                    "client_secret": client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": cls.REDIRECT_URI,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        token_file = get_token_path("tiktok", account)
        with token_file.open("w") as f:
            json.dump(data, f, indent=2)

        logger.success(f"[TikTok OAuth] Saved token to {token_file}")
        return await cls.verify(account)

    @classmethod
    async def verify(cls, account: str = "default") -> bool:
        token_file = get_token_path("tiktok", account)
        if not token_file.exists():
            return False
        try:
            with token_file.open() as f:
                token_data = json.load(f)
            token = token_data.get("access_token") or token_data.get("data", {}).get("access_token")

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://open.tiktokapis.com/v2/user/info/",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"fields": "display_name,avatar_url"},
                )
                if resp.status_code == 200:
                    d = resp.json()
                    name = d.get("data", {}).get("user", {}).get("display_name", "TikTok User")
                    logger.success(f"[TikTok OAuth] [{account}] Connected: '{name}'")
                    return True
        except Exception as exc:
            logger.warning(f"[TikTok OAuth] [{account}] Verification failed: {exc}")
        return False


# ── Kick Status & Verification ────────────────────────────────────────────────

class KickAuth:
    """Kick public API connection verifier (no user OAuth required)."""

    @classmethod
    async def verify(cls) -> bool:
        """Verify Kick API reachability."""
        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
                timeout=12.0,
                follow_redirects=True,
            ) as client:
                resp = await client.get("https://kick.com/api/v2/categories")
                # 200 is healthy, 403 indicates reachable behind Cloudflare challenge
                return resp.status_code in (200, 403)
        except Exception as exc:
            logger.warning(f"[Kick Auth] Reachability check warning: {exc}")
            return False


# ── Global Status Checker (Multi-Account Aware) ──────────────────────────────

async def get_all_auth_status() -> list[dict[str, Any]]:
    """Return status of all configured accounts across all platforms."""
    cfg = load_accounts_config()
    accounts_cfg = cfg.get("accounts", {})

    status_list: list[dict[str, Any]] = []

    # 1. YouTube Accounts
    yt_accounts = accounts_cfg.get("youtube_shorts", [{"id": "default", "name": "Default YouTube"}])
    for acc in yt_accounts:
        acc_id = acc.get("id", "default")
        token_path = Path(acc.get("credentials_file") or get_token_path("youtube", acc_id))
        is_verified = YouTubeAuth.verify(acc_id)
        status_list.append({
            "platform": "YouTube Shorts",
            "account_id": acc_id,
            "account_name": acc.get("name", acc_id),
            "configured": token_path.exists(),
            "verified": is_verified,
            "token_path": str(token_path),
        })

    # 2. TikTok Accounts
    tt_accounts = accounts_cfg.get("tiktok", [{"id": "default", "name": "Default TikTok"}])
    for acc in tt_accounts:
        acc_id = acc.get("id", "default")
        token_path = Path(acc.get("credentials_file") or get_token_path("tiktok", acc_id))
        is_verified = await TikTokAuth.verify(acc_id)
        status_list.append({
            "platform": "TikTok",
            "account_id": acc_id,
            "account_name": acc.get("name", acc_id),
            "configured": token_path.exists(),
            "verified": is_verified,
            "token_path": str(token_path),
        })

    # 3. Instagram Accounts
    ig_accounts = accounts_cfg.get("instagram_reels", [{"id": "default", "name": "Default Instagram"}])
    for acc in ig_accounts:
        acc_id = acc.get("id", "default")
        token_path = Path(acc.get("credentials_file") or get_token_path("instagram", acc_id))
        is_verified = await InstagramAuth.verify(acc_id)
        status_list.append({
            "platform": "Instagram Reels",
            "account_id": acc_id,
            "account_name": acc.get("name", acc_id),
            "configured": token_path.exists(),
            "verified": is_verified,
            "token_path": str(token_path),
        })

    # 4. Twitch (Ingest Source)
    twitch_token = get_token_path("twitch", "default")
    status_list.append({
        "platform": "Twitch (Ingest)",
        "account_id": "default",
        "account_name": "Twitch Streamer/Source",
        "configured": twitch_token.exists() or bool(get_settings().twitch_client_id),
        "verified": await TwitchAuth.verify("default"),
        "token_path": str(twitch_token),
    })

    # 5. Kick (Ingest Source)
    status_list.append({
        "platform": "Kick (Ingest)",
        "account_id": "public",
        "account_name": "Public API",
        "configured": True,
        "verified": True,
        "token_path": "N/A (Public API)",
    })

    return status_list
