"""Universal hashtag and categorization engine for Shorts, TikTok, and Reels."""

from __future__ import annotations

import random
from typing import Optional


CATEGORY_TAGS: dict[str, list[str]] = {
    "gta6": [
        "gta6", "gtavi", "grandtheftauto6", "vicecity", "gta6clips",
        "gta6gameplay", "gta6leaks", "gta6trailer", "rockstargames", "gta",
    ],
    "basketball": [
        "nba", "basketball", "nbahighlights", "dunk", "poster",
        "hoops", "crossover", "buzzerbeater", "lebron", "curry",
    ],
    "cricket": [
        "cricket", "crickethighlights", "ipl", "wicket", "six",
        "cleanbowled", "cricketlover", "teamindia", "viratkohli", "bcci",
    ],
    "football": [
        "football", "soccer", "championsleague", "premierleague", "goals",
        "screamer", "skills", "ronaldo", "messi", "fifa",
    ],
    "combat": [
        "ufc", "mma", "boxing", "knockout", "ko", "ufchighlights",
        "conormcgregor", "danawhite", "fightnight",
    ],
    "cartoons": [
        "tomandjerry", "cartoons", "animation", "classiccartoons",
        "funnycartoons", "nostalgia", "childhoodmemories",
    ],
    "movies": [
        "movies", "movieclips", "film", "cinema", "movietime",
        "movierecap", "hollywood", "mustwatch",
    ],
    "general": [
        "gaming", "gamer", "shorts", "viral", "fyp", "foryou",
        "trending", "epicmoments", "twitchclips", "streamerfails",
    ],
}


def get_hashtags(category: str = "general", count: int = 12) -> list[str]:
    """Return a curated mix of primary, category, and viral tags."""
    cat_key = category.lower()
    if "gta" in cat_key:
        cat_key = "gta6"
    elif "nba" in cat_key or "basket" in cat_key:
        cat_key = "basketball"
    elif "cricket" in cat_key:
        cat_key = "cricket"
    elif "foot" in cat_key or "soccer" in cat_key:
        cat_key = "football"
    elif "combat" in cat_key or "ufc" in cat_key:
        cat_key = "combat"
    elif "cartoon" in cat_key or "anim" in cat_key:
        cat_key = "cartoons"
    elif "movie" in cat_key:
        cat_key = "movies"
    else:
        cat_key = "general"

    pool = set(CATEGORY_TAGS.get("general", []))
    pool.update(CATEGORY_TAGS.get(cat_key, []))

    selected = random.sample(list(pool), min(count, len(pool)))
    return [t.lower() for t in selected]


def format_hashtag_string(tags: list[str]) -> str:
    """Format list of tag words into #tag string."""
    return " ".join(f"#{t.strip('#')}" for t in tags)
