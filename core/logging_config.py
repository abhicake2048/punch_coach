"""Low-overhead rotating log configuration for CornerCoach."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOGGER_NAME = "cornercoach"


def configure_logging(
    log_directory: str | Path | None = None,
    *,
    console: bool = False,
) -> Path:
    """Configure the shared CornerCoach logger and return the active log path.

    Configuration is idempotent, which is important because Streamlit reruns
    the application module. A small rotating log prevents long coaching
    sessions from consuming unbounded disk space.
    """
    directory = (
        Path(log_directory)
        if log_directory is not None
        else Path(__file__).resolve().parents[1] / "logs"
    )
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "cornercoach.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not any(getattr(handler, "_cornercoach_file", False) for handler in logger.handlers):
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler._cornercoach_file = True  # type: ignore[attr-defined]
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    if console and not any(
        getattr(handler, "_cornercoach_console", False) for handler in logger.handlers
    ):
        console_handler = logging.StreamHandler()
        console_handler._cornercoach_console = True  # type: ignore[attr-defined]
        console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(console_handler)

    return log_path
