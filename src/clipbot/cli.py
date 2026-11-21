"""ClipBot CLI — the unified command line interface."""

from __future__ import annotations

import asyncio
from typing import Optional

import aiosqlite
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from clipbot.core.config import get_config
from clipbot.core.db import init_db
from clipbot.core.log import setup_logging
from clipbot.core.storage import StorageManager
from clipbot.scoring.tune import tune_weights_from_labels

app = typer.Typer(
    name="clipbot",
    help="🎬 Universal Highlight Clipper & Episodic Publisher — Gaming (GTA 6), Sports, Movies & Cartoons",
    add_completion=False,
)
auth_app = typer.Typer(help="🔐 OAuth authentication for YouTube, Twitch, Instagram, TikTok & Kick")
app.add_typer(auth_app, name="auth")

console = Console()


# ── Auth Commands ─────────────────────────────────────────────────────────────

@auth_app.command("status")
def auth_status() -> None:
    """Check OAuth connection status for all accounts across all platforms."""
    from clipbot.core.auth import get_all_auth_status

    status_list = asyncio.run(get_all_auth_status())
    table = Table(title="🔐 Multi-Account Platform Authentication Status", show_header=True, header_style="bold cyan")
    table.add_column("Platform", style="bold")
    table.add_column("Account ID", style="yellow")
    table.add_column("Account Name", style="white")
    table.add_column("Configured", style="cyan")
    table.add_column("Connected / Verified", style="bold")
    table.add_column("Token Path", style="dim")

    for item in status_list:
        conf_str = "✅ Yes" if item["configured"] else "❌ No"
        ver_str = "[bold green]✅ Connected[/bold green]" if item["verified"] else "[yellow]⚠ Not Verified[/yellow]"
        table.add_row(item["platform"], item["account_id"], item["account_name"], conf_str, ver_str, item["token_path"])

    console.print(table)


@auth_app.command("youtube")
def auth_youtube(
    secrets: Optional[str] = typer.Option(None, help="Path to Google client_secrets_*.json file"),
    account: str = typer.Option("default", "--account", "-a", help="Account identifier (e.g. yt_main, yt_secondary)"),
) -> None:
    """Run interactive Google OAuth flow for a specific YouTube channel."""
    from clipbot.core.auth import YouTubeAuth
    console.print(f"[bold cyan]🔐 Starting YouTube OAuth flow for account: '{account}'...[/bold cyan]")
    ok = YouTubeAuth.authenticate(secrets, account=account)
    if ok:
        console.print(f"[bold green]✅ YouTube account '{account}' authorized successfully![/bold green]")
    else:
        console.print(f"[bold red]❌ YouTube authorization failed for '{account}'.[/bold red]")


@auth_app.command("twitch")
def auth_twitch(
    account: str = typer.Option("default", "--account", "-a", help="Account identifier"),
) -> None:
    """Run interactive Twitch user authorization flow."""
    from clipbot.core.auth import TwitchAuth
    console.print(f"[bold cyan]🔐 Starting Twitch OAuth flow for account: '{account}'...[/bold cyan]")
    ok = asyncio.run(TwitchAuth.authenticate(account=account))
    if ok:
        console.print(f"[bold green]✅ Twitch account '{account}' authorized![/bold green]")
    else:
        console.print(f"[bold red]❌ Twitch authorization failed for '{account}'.[/bold red]")


@auth_app.command("instagram")
def auth_instagram(
    app_id: str = typer.Option(..., prompt=True, help="Meta / Facebook App ID"),
    app_secret: str = typer.Option(..., prompt=True, hide_input=True, help="Meta App Secret"),
    account: str = typer.Option("default", "--account", "-a", help="Account identifier (e.g. ig_main, ig_page_2)"),
) -> None:
    """Run interactive Meta OAuth flow for an Instagram Reels page."""
    from clipbot.core.auth import InstagramAuth
    console.print(f"[bold cyan]🔐 Starting Instagram OAuth flow for account: '{account}'...[/bold cyan]")
    ok = asyncio.run(InstagramAuth.authenticate(app_id, app_secret, account=account))
    if ok:
        console.print(f"[bold green]✅ Instagram account '{account}' authorized![/bold green]")
    else:
        console.print(f"[bold red]❌ Instagram authorization failed for '{account}'.[/bold red]")


@auth_app.command("tiktok")
def auth_tiktok(
    account: str = typer.Option("default", "--account", "-a", help="Account identifier (e.g. tt_main, tt_niche_2)"),
) -> None:
    """Run interactive TikTok Content Posting OAuth flow for a specific account."""
    from clipbot.core.auth import TikTokAuth
    console.print(f"[bold cyan]🔐 Starting TikTok OAuth flow for account: '{account}'...[/bold cyan]")
    ok = asyncio.run(TikTokAuth.authenticate(account=account))
    if ok:
        console.print(f"[bold green]✅ TikTok account '{account}' authorized![/bold green]")
    else:
        console.print(f"[bold red]❌ TikTok authorization failed for '{account}'.[/bold red]")


@auth_app.command("kick")
def auth_kick() -> None:
    """Verify Kick connection status."""
    from clipbot.core.auth import KickAuth
    console.print("[bold cyan]Checking Kick public endpoint...[/bold cyan]")
    ok = asyncio.run(KickAuth.verify())
    if ok:
        console.print("[bold green]✅ Kick API is reachable and operational (No OAuth required)![/bold green]")
    else:
        console.print("[bold red]❌ Kick API unreachable.[/bold red]")



# ── Pipeline & Review Commands ────────────────────────────────────────────────

@app.command()
def run(
    profile: str = typer.Option("gta_chaos", help="Scoring profile (gta_chaos, gta_rp)"),
    render: bool = typer.Option(True, "--render/--no-render", help="Automatically render 9:16 shorts"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose debug logging"),
) -> None:
    """Run one full pipeline cycle: discover -> download -> score -> render."""
    setup_logging("DEBUG" if verbose else "INFO")
    console.print(f"[bold green]🎬 Starting ClipBot cycle with profile:[/bold green] [bold yellow]{profile}[/bold yellow]")
    from clipbot.pipeline import run_cycle
    asyncio.run(run_cycle(profile_name=profile, auto_render=render))


@app.command()
def daemon(
    profile: str = typer.Option("gta_chaos", help="Scoring profile"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run ClipBot as an automated background daemon (scheduled cycles + publishing)."""
    setup_logging("DEBUG" if verbose else "INFO")
    console.print(f"[bold cyan]⏰ Starting ClipBot background scheduler daemon...[/bold cyan]")
    from clipbot.scheduler import run_daemon
    try:
        asyncio.run(run_daemon(profile=profile))
    except KeyboardInterrupt:
        console.print("[yellow]Daemon stopped by user[/yellow]")


@app.command()
def review(
    host: str = typer.Option("127.0.0.1", help="Host interface"),
    port: int = typer.Option(8000, help="Web port"),
) -> None:
    """Launch the interactive web review dashboard."""
    setup_logging()
    console.print(f"[bold cyan]🖥  Review Dashboard running at:[/bold cyan] [bold yellow]http://{host}:{port}[/bold yellow]")
    import uvicorn
    from clipbot.review.app import create_app
    uvicorn.run(create_app(), host=host, port=port)


@app.command()
def publish(
    dry_run: bool = typer.Option(True, "--dry-run/--live", help="Simulate or perform actual upload"),
    platform: Optional[str] = typer.Option(None, help="Target specific platform (youtube_shorts, tiktok, instagram_reels)"),
    account: Optional[str] = typer.Option(None, "--account", "-a", help="Target specific account ID (e.g. yt_main, ig_page_2)"),
    limit: int = typer.Option(3, help="Max approved clips to publish"),
) -> None:
    """Publish approved clips across all accounts or a targeted account."""
    setup_logging()
    config = get_config()
    config.publish.dry_run = dry_run

    if dry_run:
        console.print("[bold yellow]⚠  Dry-run mode active — simulated upload (no actual posts)[/bold yellow]")
    else:
        console.print("[bold red]🚀 LIVE mode — uploading to social accounts[/bold red]")

    from clipbot.publish.manager import PublishManager
    mgr = PublishManager(config.database.path)
    asyncio.run(mgr.dispatch_approved_clips(limit=limit, platform_filter=platform, account_filter=account))


@app.command(name="clip-url")
def clip_url(
    url: str = typer.Argument(..., help="URL of stream or video (Twitch, Kick, YouTube)"),
    profile: str = typer.Option("general_viral", "--profile", "-p", help="Scoring profile"),
    max_clips: int = typer.Option(3, "--max-clips", "-c", help="Number of highlights to extract and render"),
    layout: Optional[str] = typer.Option(None, "--layout", "-l", help="Layout override: 'split_stack' or 'blur_canvas' (default: auto)"),
) -> None:
    """Run full highlights pipeline on a target URL: download -> detect -> score -> render 9:16 vertical shorts."""
    setup_logging("DEBUG")
    console.print(f"[bold cyan]🎬 Processing highlights on URL:[/bold cyan] {url} [dim](max {max_clips} clips, layout={layout or 'auto'})[/dim]")
    from clipbot.pipeline import process_url_pipeline
    asyncio.run(process_url_pipeline(url, profile_name=profile, max_clips=max_clips, layout=layout))


@app.command()
def split_movie(
    input_source: str = typer.Argument(..., help="Path to movie file or streaming/YouTube URL"),
    title: str = typer.Option("Untitled Movie", "--title", "-t", help="Movie display title"),
    max_parts: Optional[int] = typer.Option(None, "--max-parts", "-m", help="Limit parts generated (e.g. for testing)"),
) -> None:
    """Chop a full-length movie into a sequential series of 9:16 vertical shorts (Part 1, Part 2...)."""
    setup_logging("INFO")
    console.print(f"[bold cyan]🍿 Starting Movie Episodic Splitter for:[/bold cyan] [bold yellow]{title}[/bold yellow]")
    from clipbot.render.movie_splitter import MovieSplitter
    splitter = MovieSplitter()
    clips = asyncio.run(splitter.split_and_queue_movie(input_source, title=title, max_parts=max_parts))
    console.print(f"[bold green]✅ Generated {len(clips)} sequential episodes! Open 'clipbot review' to preview.[/bold green]")


@app.command()
def status() -> None:
    """Show database statistics, pending review counts, and cache size."""
    config = get_config()
    db_path = config.database.path

    async def _get_status():
        await init_db(db_path)
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM assets") as c:
                total_assets = (await c.fetchone())[0]
            async with db.execute("SELECT COUNT(*) FROM clips WHERE status IN ('rendered', 'candidate')") as c:
                pending_clips = (await c.fetchone())[0]
            async with db.execute("SELECT COUNT(*) FROM clips WHERE status='approved'") as c:
                approved_clips = (await c.fetchone())[0]
            async with db.execute("SELECT COUNT(*) FROM clips WHERE status='published'") as c:
                published_clips = (await c.fetchone())[0]
            async with db.execute("SELECT COUNT(*) FROM posts WHERE status='published'") as c:
                total_posts = (await c.fetchone())[0]

        storage = StorageManager()
        cache_mb = storage.get_media_size_bytes() / (1024 ** 2)

        table = Table(title="🎬 ClipBot Pipeline Status", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold green")

        table.add_row("Discovered Assets", str(total_assets))
        table.add_row("Pending Review Clips", str(pending_clips))
        table.add_row("Approved Clips", str(approved_clips))
        table.add_row("Published Clips", str(published_clips))
        table.add_row("Total Social Posts", str(total_posts))
        table.add_row("Raw Media Cache", f"{cache_mb:.1f} MB / {storage.max_cache_gb} GB")

        console.print(table)

    asyncio.run(_get_status())


@app.command()
def tune(
    profile: str = typer.Option("gta_chaos", help="Profile to tune"),
    save: bool = typer.Option(True, help="Save tuned weights to YAML profile"),
) -> None:
    """Analyze approved/rejected review labels and re-fit signal weights."""
    setup_logging()
    console.print(f"[bold cyan]🎯 Tuning signal weights from review labels for profile: {profile}[/bold cyan]")
    config = get_config()
    weights = asyncio.run(tune_weights_from_labels(config.database.path, profile, save=save))
    console.print(f"[bold green]Updated Weights:[/bold green] {weights}")


@app.command()
def cleanup() -> None:
    """Manually prune raw media cache files."""
    setup_logging()
    storage = StorageManager()
    count = storage.prune_media_cache()
    console.print(f"[bold green]Cleaned up {count} cached media files.[/bold green]")


if __name__ == "__main__":
    app()
