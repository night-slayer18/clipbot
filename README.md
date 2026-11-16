# ClipBot Studio 🎬

**Universal AI Highlight Clipping, Episodic Movie Splitting & Publishing Automation Engine** for **Gaming (GTA 6, etc.), Sports (Basketball, Cricket, Football), Movies & Cartoons** across **Twitch, Kick, and YouTube**.

Extracts high-impact moments, reframes into high-retention 9:16 vertical shorts with animated karaoke subtitles, and queues them for scheduled publishing to YouTube Shorts, TikTok, and Instagram Reels.

---

## Quick Start

```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Check connection & OAuth status
clipbot auth status

# 3. Run a discovery & render cycle with any profile:
clipbot run --profile gta6_chaos         # GTA 6 Vice City action
clipbot run --profile basketball_nba     # NBA highlights
clipbot run --profile cricket            # Cricket wickets & sixes
clipbot run --profile football_soccer    # Football goals & skills
clipbot run --profile animation_cartoon  # Slapstick cartoons (Tom & Jerry)
clipbot run --profile general_viral      # Universal streamer/gaming clips

# 4. Or split a full movie into a sequential vertical series:
clipbot split-movie /path/to/movie.mp4 --title "Inception"

# 5. Review & trim clips in browser
clipbot review --port 8000
# Open http://127.0.0.1:8000

# 6. Publish approved clips
clipbot publish --dry-run   # or --live to post
```

---

## 🔐 OAuth 2.0 Setup For All Platforms

ClipBot features interactive OAuth 2.0 authentication commands that open your browser, exchange tokens, and save credentials to `data/credentials/`:

### 1. YouTube Shorts (Google OAuth 2.0)
1. Go to [Google Cloud Console](https://console.cloud.google.com/) → Enable **YouTube Data API v3**.
2. Create Credentials → **OAuth client ID** → Application type: **Desktop app**.
3. Download the JSON client secret file.
4. Run:
```bash
clipbot auth youtube --secrets /path/to/client_secret_*.json
```

### 2. Twitch OAuth
1. Go to [Twitch Developer Console](https://dev.twitch.tv/console).
2. Register an app with OAuth Redirect URL: `http://127.0.0.1:8765/callback/twitch`.
3. Add `CLIPBOT_TWITCH_CLIENT_ID` and `CLIPBOT_TWITCH_CLIENT_SECRET` in `.env`.
4. Run:
```bash
clipbot auth twitch
```

### 3. Instagram Reels (Meta Graph API)
1. Create a Meta Developer App at [developers.facebook.com](https://developers.facebook.com/).
2. Add Instagram Graph API & set Redirect URL: `http://127.0.0.1:8765/callback/instagram`.
3. Run:
```bash
clipbot auth instagram --app-id <YOUR_APP_ID> --app-secret <YOUR_APP_SECRET>
```

### 4. TikTok (Content Posting API)
1. Register at [developers.tiktok.com](https://developers.tiktok.com/) and request Content Posting API.
2. Set Redirect URL: `http://127.0.0.1:8765/callback/tiktok`.
3. Add `CLIPBOT_TIKTOK_CLIENT_KEY` and `CLIPBOT_TIKTOK_CLIENT_SECRET` in `.env`.
4. Run:
```bash
clipbot auth tiktok
```

### 5. Kick
Kick uses public discovery endpoints — no OAuth application registration is needed! Verify anytime:
```bash
clipbot auth kick
```

---

## CLI Commands Reference

| Command | Description |
|---|---|
| `clipbot auth status` | Check connection and token validity for all platforms |
| `clipbot auth youtube` | Run Google OAuth flow for YouTube Shorts |
| `clipbot auth twitch` | Run Twitch OAuth user authorization |
| `clipbot auth instagram` | Run Meta OAuth flow for Instagram Reels |
| `clipbot auth tiktok` | Run TikTok Content Posting OAuth |
| `clipbot auth kick` | Verify Kick public API access |
| `clipbot run --profile <name>` | Run discover -> download -> score -> 9:16 render cycle |
| `clipbot split-movie <url_or_file>` | Chop a full movie into a sequential Part 1, Part 2... 9:16 series |
| `clipbot daemon` | Continuous background daemon (scheduled runs + publishing) |
| `clipbot review` | Launch interactive SaaS web review UI |
| `clipbot publish` | Dispatch approved clips to social platforms |
| `clipbot test-clip <url>` | Test pipeline on any video or clip URL |
| `clipbot status` | View database stats and cache size |
| `clipbot tune` | Re-fit signal weights based on your approved/rejected labels |
| `clipbot cleanup` | Clean cached media files |

---

## Available Profiles

* `gta6_chaos`: GTA 6 (Vice City / Leonida) high-speed pursuits, shootouts, stunts, and "WASTED" / "BUSTED" moments.
* `basketball_nba`: NBA fastbreaks, poster dunks, crossovers, buzzer beaters, and rim-shaking action.
* `cricket`: Wickets, middle stump clean bowled, towering sixes, diving catches, and super overs.
* `football_soccer`: Screamers, top-corner goals, solo runs, red cards, and last-minute winners.
* `combat_ufc`: One-punch knockouts, head kicks, fight-ending submissions, and canvas thuds.
* `animation_cartoon`: Slapstick physical comedy, sound effect crashes, hammer hits, and chase scenes (Tom & Jerry, Looney Tunes).
* `movie_episodic`: Long-form movie series split into sequential ~58s parts with dialogue silence boundary cuts.
* `general_viral`: Universal viral streamer and gaming moments.
