"""Bar loaders. CSV today; broker/REST adapters plug in behind the same ABC."""

from __future__ import annotations

from xaubot.data.loaders.base import BarLoader
from xaubot.data.loaders.csv_loader import CsvBarLoader

__all__ = ["BarLoader", "CsvBarLoader"]
