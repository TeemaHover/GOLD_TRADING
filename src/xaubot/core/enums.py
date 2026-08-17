"""Enumerations shared across the whole system.

Everything that has a fixed, closed set of values lives here so that no module
invents its own string constants. String enums are used throughout so that
values round-trip cleanly through YAML, JSON, and Parquet.
"""

from __future__ import annotations

from datetime import timedelta
from enum import StrEnum


class Timeframe(StrEnum):
    """Supported bar timeframes.

    The base (execution) timeframe is :attr:`M5`. All other timeframes are
    derived from it by resampling and are used strictly as *context*.
    """

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def minutes(self) -> int:
        """Duration of one bar in minutes."""
        return _TF_MINUTES[self]

    @property
    def duration(self) -> timedelta:
        """Duration of one bar as a :class:`~datetime.timedelta`."""
        return timedelta(minutes=self.minutes)

    @property
    def pandas_freq(self) -> str:
        """Pandas offset alias used for resampling."""
        return _TF_FREQ[self]

    @property
    def bars_per_day(self) -> int:
        """Number of bars in a 24-hour period (calendar, not trading, day)."""
        return max(1, 1440 // self.minutes)

    def is_intraday(self) -> bool:
        """True for timeframes shorter than one day."""
        return self.minutes < 1440


_TF_MINUTES: dict[Timeframe, int] = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
}

_TF_FREQ: dict[Timeframe, str] = {
    Timeframe.M1: "1min",
    Timeframe.M5: "5min",
    Timeframe.M15: "15min",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1D",
}


class TimestampConvention(StrEnum):
    """Whether a source file stamps a bar with its open time or its close time.

    Getting this wrong shifts every feature by one bar and is a silent,
    catastrophic source of look-ahead bias, so it is always explicit in config
    and never inferred.
    """

    OPEN = "open"
    CLOSE = "close"


class Session(StrEnum):
    """Trading sessions. ``OFF`` covers the daily maintenance break/weekend."""

    ASIA = "ASIA"
    LONDON = "LONDON"
    NY = "NY"
    LONDON_NY_OVERLAP = "LONDON_NY_OVERLAP"
    OFF = "OFF"


class Side(StrEnum):
    """Direction of an open position."""

    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        """+1 for long, -1 for short. Used to make P&L math direction-agnostic."""
        return 1 if self is Side.LONG else -1


class SignalAction(StrEnum):
    """Output of the signal engine. ``WAIT`` is expected to dominate."""

    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"


class Regime(StrEnum):
    """Market regime labels (see docs/ARCHITECTURE.md section 5.3)."""

    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"


class VolRegime(StrEnum):
    """Volatility bucket derived from the trailing ATR percentile."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class BarrierOutcome(StrEnum):
    """Result of a triple-barrier evaluation."""

    TP_HIT = "TP_HIT"
    SL_HIT = "SL_HIT"
    TIMEOUT = "TIMEOUT"


class ExitReason(StrEnum):
    """Why a position was closed."""

    STOP_LOSS = "SL"
    TAKE_PROFIT_1 = "TP1"
    TAKE_PROFIT_2 = "TP2"
    TAKE_PROFIT_3 = "TP3"
    TRAILING_STOP = "TRAIL"
    TIME_STOP = "TIME"
    END_OF_DAY = "EOD"
    RISK_HALT = "RISK_HALT"
    MANUAL = "MANUAL"


class TradingMode(StrEnum):
    """Execution context. ``LIVE`` remains unimplemented until the go/no-go
    checklist in docs/RUNBOOK.md is satisfied."""

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class DataIssue(StrEnum):
    """Categories of problem the data validator can report."""

    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    NON_MONOTONIC = "non_monotonic"
    MISSING_BAR = "missing_bar"
    NULL_VALUE = "null_value"
    NON_POSITIVE_PRICE = "non_positive_price"
    OHLC_INCONSISTENT = "ohlc_inconsistent"
    NEGATIVE_VOLUME = "negative_volume"
    ZERO_VOLUME = "zero_volume"
    EXTREME_RETURN = "extreme_return"
    FLAT_BAR = "flat_bar"
    OFF_GRID_TIMESTAMP = "off_grid_timestamp"


class Severity(StrEnum):
    """How a data issue is treated by the pipeline."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
