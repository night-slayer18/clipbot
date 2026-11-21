"""FastAPI review dashboard — interactive web UI to review, trim, and approve clips."""

from __future__ import annotations

import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiosqlite
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from clipbot.core.config import get_config, get_settings
from clipbot.core.db import init_db
from clipbot.core.repository import get_clips_for_review, set_clip_decision
from clipbot.render.compositor import VideoCompositor


TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
security = HTTPBasic(auto_error=False)


class ApproveRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    in_point_delta: float = 0.0
    out_point_delta: float = 0.0


class RejectRequest(BaseModel):
    reason: Optional[str] = ""


class ReRenderRequest(BaseModel):
    start_time: float
    end_time: float
    layout: Optional[str] = None


def create_app() -> FastAPI:
    config = get_config()
    settings = get_settings()
    db_path = config.database.path

    # Security: HTTP Basic Auth gate when review password is configured
    auth_user = os.environ.get("CLIPBOT_REVIEW_USER") or getattr(settings, "review_user", "admin")
    auth_password = os.environ.get("CLIPBOT_REVIEW_PASSWORD") or getattr(settings, "review_password", "")
    auth_enabled = bool(auth_password)

    def verify_auth(credentials: Optional[HTTPBasicCredentials] = Depends(security)):
        if not auth_enabled:
            return True
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Basic"},
            )
        correct_username = secrets.compare_digest(credentials.username, auth_user)
        correct_password = secrets.compare_digest(credentials.password, auth_password)
        if not (correct_username and correct_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        return True

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_db(db_path)
        yield

    app = FastAPI(
        title="ClipBot Review Dashboard",
        version="1.0.0",
        lifespan=lifespan,
        dependencies=[Depends(verify_auth)],
    )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR)) if TEMPLATES_DIR.exists() else None

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        if templates:
            return templates.TemplateResponse("index.html", {"request": request})
        return HTMLResponse("<h1>ClipBot Review UI</h1>")

    @app.get("/api/stats")
    async def get_stats():
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM assets") as c:
                total_assets = (await c.fetchone())[0]
            async with db.execute("SELECT COUNT(*) FROM clips WHERE status IN ('rendered', 'candidate')") as c:
                pending_clips = (await c.fetchone())[0]
            async with db.execute("SELECT COUNT(*) FROM clips WHERE status='approved'") as c:
                approved_clips = (await c.fetchone())[0]
            async with db.execute("SELECT COUNT(*) FROM clips WHERE status='published'") as c:
                published_clips = (await c.fetchone())[0]

        return {
            "total_assets": total_assets,
            "pending_clips": pending_clips,
            "approved_clips": approved_clips,
            "published_clips": published_clips,
        }

    @app.get("/api/clips")
    async def list_clips(status: str = Query("pending")):
        async with aiosqlite.connect(db_path) as db:
            if status == "pending":
                query = """
                SELECT c.*, a.creator, a.source, a.title as asset_title, a.url as asset_url
                FROM clips c JOIN assets a ON c.asset_id = a.id
                WHERE c.status IN ('rendered', 'candidate')
                ORDER BY c.overall_score DESC LIMIT 60
                """
            elif status == "approved":
                query = """
                SELECT c.*, a.creator, a.source, a.title as asset_title, a.url as asset_url
                FROM clips c JOIN assets a ON c.asset_id = a.id
                WHERE c.status='approved' ORDER BY c.reviewed_at DESC LIMIT 60
                """
            else:
                query = """
                SELECT c.*, a.creator, a.source, a.title as asset_title, a.url as asset_url
                FROM clips c JOIN assets a ON c.asset_id = a.id
                ORDER BY c.created_at DESC LIMIT 60
                """
            async with db.execute(query) as cur:
                rows = await cur.fetchall()
                cols = [d[0] for d in cur.description]

        clips = []
        for r in rows:
            d = dict(zip(cols, r))
            d["tags"] = json.loads(d.get("tags_json") or "[]")
            d["scores_breakdown"] = json.loads(d.get("scores_breakdown") or "{}")
            clips.append(d)

        return {"clips": clips}

    @app.get("/video/{clip_id}")
    async def stream_video(clip_id: str, request: Request):
        """Byte-range capable video streaming for HTML5 video player."""
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT rendered_clip_path, raw_clip_path FROM clips WHERE id=?", (clip_id,)) as cur:
                row = await cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Clip not found")

        rendered_path, raw_path = row
        file_path = Path(rendered_path or raw_path or "")
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Video file does not exist on disk")

        file_size = file_path.stat().st_size
        range_header = request.headers.get("range")

        if not range_header:
            return FileResponse(file_path, media_type="video/mp4", filename=file_path.name)

        # Parse range header: e.g. "bytes=0-1048576"
        bytes_prefix = "bytes="
        if not range_header.startswith(bytes_prefix):
            return FileResponse(file_path, media_type="video/mp4")

        range_val = range_header[len(bytes_prefix):].strip()
        parts = range_val.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else min(start + 1024 * 1024, file_size - 1)

        chunk_size = (end - start) + 1

        def iter_chunk():
            with file_path.open("rb") as f:
                f.seek(start)
                yield f.read(chunk_size)

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(iter_chunk(), status_code=206, headers=headers)

    @app.post("/api/clips/{clip_id}/approve")
    async def approve_clip(clip_id: str, req: ApproveRequest = ApproveRequest()):
        async with aiosqlite.connect(db_path) as db:
            updates = ["status='approved'", "reviewed_at=CURRENT_TIMESTAMP"]
            params = []
            if req.title:
                updates.append("title=?")
                params.append(req.title)
            if req.description:
                updates.append("description=?")
                params.append(req.description)
            if req.tags:
                updates.append("tags_json=?")
                params.append(json.dumps(req.tags))

            params.append(clip_id)
            await db.execute(f"UPDATE clips SET {', '.join(updates)} WHERE id=?", tuple(params))

            # Store label for score tuning
            await db.execute(
                """
                INSERT INTO labels (clip_id, decision, reason, in_point_delta, out_point_delta)
                VALUES (?, 'approved', '', ?, ?)
                """,
                (clip_id, req.in_point_delta, req.out_point_delta),
            )
            await db.commit()

        return {"clip_id": clip_id, "status": "approved"}

    @app.post("/api/clips/{clip_id}/reject")
    async def reject_clip(clip_id: str, req: RejectRequest = RejectRequest()):
        async with aiosqlite.connect(db_path) as db:
            await set_clip_decision(db, clip_id, "rejected", req.reason or "user_rejected")
        return {"clip_id": clip_id, "status": "rejected"}

    @app.post("/api/clips/{clip_id}/re-render")
    async def re_render_clip(clip_id: str, req: ReRenderRequest):
        """Re-render clip with adjusted in/out bounds."""
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                """
                SELECT a.local_path, c.rendered_clip_path
                FROM clips c JOIN assets a ON c.asset_id = a.id
                WHERE c.id=?
                """,
                (clip_id,),
            ) as cur:
                row = await cur.fetchone()

        if not row or not row[0]:
            raise HTTPException(status_code=404, detail="Source asset media not found")

        source_path = Path(row[0])
        out_path = Path(row[1]) if row[1] else Path(config.storage.clips_dir) / f"{clip_id}_9x16.mp4"

        compositor = VideoCompositor()
        new_path = await compositor.render_short(
            source_media=source_path,
            start_time=req.start_time,
            end_time=req.end_time,
            output_path=out_path,
            force_layout=req.layout,
        )

        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                """
                UPDATE clips SET start_time=?, end_time=?, rendered_clip_path=?, status='rendered'
                WHERE id=?
                """,
                (req.start_time, req.end_time, str(new_path), clip_id),
            )
            await db.commit()

        return {"clip_id": clip_id, "status": "rendered", "path": str(new_path)}

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "clipbot-review"}

    return app
