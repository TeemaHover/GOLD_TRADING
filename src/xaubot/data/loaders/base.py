"""Loader interface.

Every data source -- CSV today, a broker REST endpoint later -- produces the
same canonical frame, so nothing downstream needs to know where bars came from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from xaubot.core.enums import Timeframe


class BarLoader(ABC):
    """Produces a raw bar frame in canonical column form.

    Implementations are responsible for *parsing only*: column naming,
    timestamp interpretation, and timezone normalisation. They must not sort,
    deduplicate, or repair data -- that is the cleaning stage's job, and keeping
    it separate is what allows the quality report to describe what was actually
    wrong with the source.
    """

    timeframe: Timeframe
    symbol: str

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return a frame with ``open_time``, ``close_time``, OHLCV columns.

        The frame is UTC and may still contain duplicates, out-of-order rows,
        nulls, and malformed bars.
        """

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of the source, recorded in the report."""
