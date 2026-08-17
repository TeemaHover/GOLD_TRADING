"""Centralised logging setup.

Console output is human-readable via ``rich``; file output is plain text so it
greps cleanly. Every training/backtest run writes its own log file next to its
artifacts, which is what makes a result reproducible after the fact.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from rich.logging import RichHandler

_CONFIGURED = False

_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    level: str | int | None = None,
    log_file: Path | None = None,
    *,
    force: bool = False,
) -> None:
    """Configure root logging exactly once per process.

    Args:
        level: Log level name or number. Falls back to the ``XAUBOT_LOG_LEVEL``
            environment variable, then to ``INFO``.
        log_file: Optional path for a plain-text log file. Parent directories
            are created if needed.
        force: Reconfigure even if logging was already set up. Used by tests.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved = level if level is not None else os.environ.get("XAUBOT_LOG_LEVEL", "INFO")

    root = logging.getLogger()
    root.setLevel(resolved)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        omit_repeated_times=False,
        console=None,
    )
    console.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        root.addHandler(file_handler)

    # Third-party noise suppression.
    for noisy in ("matplotlib", "numexpr", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging has been configured."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)


def excepthook_to_log() -> None:
    """Route uncaught exceptions through logging so they land in the log file."""

    def _hook(exc_type, exc, tb):  # type: ignore[no-untyped-def]
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logging.getLogger("xaubot").critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _hook
