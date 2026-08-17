"""Data layer: loading, validation, cleaning, resampling, storage."""

from __future__ import annotations

from xaubot.data.cleaning import CleaningResult, clean_bars
from xaubot.data.pipeline import IngestResult, build_calendar, ingest
from xaubot.data.resample import resample_bars, resample_many
from xaubot.data.store import read_bars, store_summary, write_bars
from xaubot.data.validators import detect_gaps, detect_issues, validate_schema

__all__ = [
    "CleaningResult",
    "IngestResult",
    "build_calendar",
    "clean_bars",
    "detect_gaps",
    "detect_issues",
    "ingest",
    "read_bars",
    "resample_bars",
    "resample_many",
    "store_summary",
    "validate_schema",
    "write_bars",
]
