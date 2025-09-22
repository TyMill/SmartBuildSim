"""Utilities for configuring SmartBuildSim logging."""

from __future__ import annotations

import logging
import logging.config
from copy import deepcopy
from typing import Any

DEFAULT_LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        }
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}


def setup_logging(level: str = "INFO", config: dict[str, Any] | None = None) -> None:
    """Configure logging for the application."""

    merged = deepcopy(config or DEFAULT_LOGGING_CONFIG)
    merged.setdefault("handlers", {}).setdefault("console", {})
    merged.setdefault("root", {})
    merged["handlers"]["console"]["level"] = level.upper()
    merged["root"]["level"] = level.upper()
    logging.config.dictConfig(merged)


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured via :func:`setup_logging`."""

    return logging.getLogger(name)


__all__ = ["DEFAULT_LOGGING_CONFIG", "setup_logging", "get_logger"]