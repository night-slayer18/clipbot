"""Universal multi-niche title and description generator.

Supports:
  - GTA 6 / Grand Theft Auto VI
  - Sports (Basketball / NBA, Cricket, Football / Soccer, Combat / UFC)
  - Movies & Episodic Series
  - Animation & Slapstick Cartoons
  - General Gaming & IRL Streamers
"""

from __future__ import annotations

import random
from clipbot.core.models import Asset, Clip
from clipbot.meta.hashtags import format_hashtag_string, get_hashtags


TITLE_BANKS: dict[str, list[str]] = {
    "gta6": [
        "🔥 GTA 6 LEAKS & CHAOS: {creator} in Vice City!",
        "💀 GTA 6 POLICE PURSUIT ENDS IN DISASTER | {creator}",
        "😱 This GTA 6 moment has me DEAD 💀 #gtavi",
        "🚨 BUSTED in Vice City! GTA 6 {creator} moment",
        "Bro really tried to escape the cops in GTA 6 💀 | {creator}",
        "NO WAY THIS JUST HAPPENED IN GTA 6 😂 | {creator}",
        "⚡ GTA 6 Gameplay is unreal! {creator} moment",
        "🔥 Funniest GTA 6 clip of the day | {creator}",
    ],
    "basketball": [
        "🏀 POSTER! {creator} with the insane NBA slam!",
        "😱 ANKLES BROKEN! Unbelievable crossover | NBA",
        "🔥 FOR THE WIN! Clutch buzzer beater moment 🏀",
        "💀 He put him on a poster! Insane dunk #nba",
        "BANG! Unbelievable shot in the clutch #basketball",
    ],
    "cricket": [
        "🏏 CLEAN BOWLED! Middle stump completely shattered!",
        "😱 UNBELIEVABLE CATCH! Fielder dives out of nowhere!",
        "💥 THAT IS OUT OF THE GROUND! Monster six #cricket",
        "🔥 Hat-trick moment! Absolute bowling masterclass 🏏",
        "WHAT A THROW! Direct hit run-out in the final over!",
    ],
    "football": [
        "⚽ WHAT A GOAL! Absolute screamer into the top corner!",
        "😱 UNREAL SKILLS! The defender got sent to the shops 💀",
        "🔥 LAST MINUTE WINNER! Stadium goes absolutely wild!",
        "WHAT A SAVE! Goalkeeper with lightning reflexes ⚽",
        "SIUUU! Unstoppable strike from outside the box #football",
    ],
    "combat": [
        "🥊 OUT COLD! Clean one-punch knockout in the octagon!",
        "😱 HE'S HURT! Wild exchange ends in dramatic KO #ufc",
        "⚡ UNBELIEVABLE HEAD KICK! Fight ends in seconds 🥊",
        "GOODNIGHT! Brutal finish in the main event #mma",
    ],
    "cartoons": [
        "🐱 Tom and Jerry: When the trap backfires completely 😂",
        "💀 Classic slapstick moment that never gets old #nostalgia",
        "🤣 Tom thought he had him this time... #tomandjerry",
        "Classic animation chaos that made our childhood 😂",
    ],
    "movies": [
        "🍿 THIS MOVIE SCENE GAVE ME CHILLS 😱",
        "The most intense cinema scene ever filmed 🎬",
        "Wait till the end of this scene... 🍿",
        "Unbelievable cinema masterpiece 🎬",
        "Bro wasn't ready for what happened next 💀",
    ],
    "general": [
        "😂 You won't believe what just happened to {creator}...",
        "💀 Bro really thought he would get away with this...",
        "🔥 Funniest streamer moment of the week | {creator}",
        "😱 WAIT TILL THE END! You won't expect this #viral",
        "NO WAY THIS IS REAL 😂 | {creator}",
    ],
}


def detect_niche(asset: Asset, profile_name: str = "") -> str:
    """Identify content niche from asset title, game, or profile."""
    combined = f"{asset.title} {asset.game} {profile_name}".lower()

    if "gta" in combined or "vice" in combined or "leonida" in combined:
        return "gta6"
    elif "nba" in combined or "basket" in combined:
        return "basketball"
    elif "cricket" in combined or "ipl" in combined:
        return "cricket"
    elif "foot" in combined or "soccer" in combined or "goal" in combined:
        return "football"
    elif "ufc" in combined or "mma" in combined or "box" in combined:
        return "combat"
    elif "cartoon" in combined or "jerry" in combined or "anim" in combined:
        return "cartoons"
    elif "movie" in combined or "film" in combined:
        return "movies"
    return "general"


def generate_title(asset: Asset, clip: Clip | None = None, profile_name: str = "") -> str:
    """Generate high-CTR title tailored to the active niche."""
    niche = detect_niche(asset, profile_name)
    templates = TITLE_BANKS.get(niche, TITLE_BANKS["general"])
    template = random.choice(templates)
    creator_name = asset.creator or "Creator"

    return template.format(creator=creator_name)


def generate_description(asset: Asset, tags: list[str] | None = None, profile_name: str = "") -> str:
    """Generate SEO-rich description with dynamic hashtags and source credit."""
    niche = detect_niche(asset, profile_name)
    creator_name = asset.creator or "the creator"
    tag_str = format_hashtag_string(tags or get_hashtags(category=niche, count=8))

    niche_label = "GTA 6" if niche == "gta6" else niche.capitalize()

    return (
        f"Insane {niche_label} moment featuring {creator_name}!\n\n"
        f"Original Creator / Streamer: {creator_name}\n"
        f"Source: {asset.url}\n\n"
        f"Subscribe and follow for daily highlights!\n\n"
        f"{tag_str} #Shorts"
    )


def generate_tags(category: str = "general", n: int = 15) -> list[str]:
    return get_hashtags(category=category, count=n)
