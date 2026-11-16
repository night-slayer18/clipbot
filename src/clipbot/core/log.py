"""Structured logging setup using Loguru."""

import sys
from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    logger.add(
        "data/logs/clipbot_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="1 day",
        retention="14 days",
        compression="gz",
    )


__all__ = ["logger", "setup_logging"]
