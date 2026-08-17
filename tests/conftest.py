"""Shared test fixtures.

Synthetic bars are generated from a seeded RNG so every test is deterministic,
and they respect the trading calendar so that gap-detection tests exercise the
real code path rather than a 24/7 grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xaubot.config.schema import (
    AppConfig,
    CsvSourceConfig,
    DataConfig,
    PathsConfig,
)
from xaubot.core.calendar import DEFAULT_CALENDAR, TradingCalendar
from xaubot.core.enums import Timeframe, TimestampConvention
from xaubot.core.time_utils import CLOSE_TIME, OPEN_TIME
from xaubot.core.types import BarFrame


def make_bars(
    n: int = 2000,
    start: str = "2026-01-05 00:00:00",
    timeframe: Timeframe = Timeframe.M5,
    seed: int = 7,
    start_price: float = 4000.0,
    calendar: TradingCalendar | None = DEFAULT_CALENDAR,
) -> pd.DataFrame:
    """Build a synthetic OHLCV frame on the given timeframe grid.

    The series is a seeded random walk with realistic gold-like volatility
    (about 0.05% per 5m bar). Bars falling outside trading hours are dropped so
    the result looks like a real feed.
    """
    rng = np.random.default_rng(seed)
    duration = pd.Timedelta(timeframe.duration)

    # Over-generate, then filter to trading hours, then take the first n.
    grid = pd.date_range(start=pd.Timestamp(start, tz="UTC"), periods=n * 3, freq=timeframe.pandas_freq)
    if calendar is not None:
        grid = grid[calendar.is_open(grid)]
    grid = grid[:n]

    steps = rng.normal(loc=0.0, scale=start_price * 0.0005, size=len(grid))
    close = start_price + np.cumsum(steps)
    open_ = np.concatenate([[start_price], close[:-1]])
    spread = np.abs(rng.normal(loc=start_price * 0.0004, scale=start_price * 0.0002, size=len(grid)))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(50, 2000, size=len(grid)).astype("float64")

    return pd.DataFrame(
        {
            OPEN_TIME: grid,
            CLOSE_TIME: grid + duration,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def bars_frame() -> pd.DataFrame:
    """Raw canonical frame (unindexed), as a loader would produce it."""
    return make_bars()


@pytest.fixture
def bar_frame(bars_frame: pd.DataFrame) -> BarFrame:
    """Validated :class:`BarFrame` of synthetic 5m bars."""
    indexed = bars_frame.set_index(CLOSE_TIME, drop=False)
    indexed.index.name = CLOSE_TIME
    return BarFrame(df=indexed, timeframe=Timeframe.M5, symbol="XAUUSD")


@pytest.fixture
def csv_path(tmp_path: Path, bars_frame: pd.DataFrame) -> Path:
    """A CSV in the same shape as the real data file."""
    target = tmp_path / "xauusd_5m.csv"
    out = bars_frame.loc[:, [OPEN_TIME, "open", "high", "low", "close", "volume"]].rename(
        columns={OPEN_TIME: "datetime"}
    )
    out.to_csv(target, index=False)
    return target


@pytest.fixture
def app_config(tmp_path: Path, csv_path: Path) -> AppConfig:
    """Minimal working config pointed at the synthetic CSV."""
    return AppConfig(
        paths=PathsConfig(data_root=tmp_path / "data", artifact_root=tmp_path / "artifacts"),
        data=DataConfig(
            source=CsvSourceConfig(
                path=csv_path,
                timeframe=Timeframe.M5,
                timestamp_column="datetime",
                timestamp_convention=TimestampConvention.OPEN,
            )
        ),
    )
