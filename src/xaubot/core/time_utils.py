"""Timestamp handling.

The whole system runs on tz-aware UTC timestamps, and every bar carries *both*
``open_time`` and ``close_time``. This is the foundation of the point-in-time
guarantee: a feature computed at decision time ``t`` may only touch rows whose
``close_time <= t``. Ambiguity about which end of a bar a timestamp refers to is
the single cheapest way to introduce a one-bar look-ahead, so it is never left
implicit.
"""

from __future__ import annotations

import pandas as pd

from xaubot.core.enums import Timeframe, TimestampConvention
from xaubot.core.errors import DataError

UTC = "UTC"

OPEN_TIME = "open_time"
CLOSE_TIME = "close_time"
OHLCV = ("open", "high", "low", "close", "volume")
BAR_COLUMNS = (OPEN_TIME, CLOSE_TIME, *OHLCV)


def ensure_utc(index: pd.DatetimeIndex | pd.Series, source_tz: str | None = None) -> pd.DatetimeIndex:
    """Return a tz-aware UTC :class:`~pandas.DatetimeIndex`.

    Args:
        index: Datetime index or series, tz-aware or naive.
        source_tz: Timezone to assume for *naive* input. If omitted, naive
            input is assumed to already be UTC.

    Raises:
        DataError: If the values cannot be interpreted as datetimes.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        tz = source_tz or UTC
        try:
            # ambiguous/nonexistent arise only for non-UTC source zones during
            # DST transitions; 'infer' handles the common ordered-series case.
            idx = idx.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        except Exception as exc:  # pandas raises several unrelated types here
            raise DataError(
                f"Could not localize naive timestamps to {tz!r}: {exc}. "
                "Fix the source timezone in config rather than guessing."
            ) from exc
    return idx.tz_convert(UTC)


def derive_bar_times(
    stamps: pd.DatetimeIndex,
    timeframe: Timeframe,
    convention: TimestampConvention,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split a single timestamp column into explicit open and close times.

    Args:
        stamps: The timestamps as they appear in the source file (UTC).
        timeframe: Bar timeframe, used for the bar duration.
        convention: Whether ``stamps`` are bar-open or bar-close times.

    Returns:
        ``(open_time, close_time)``.
    """
    duration = pd.Timedelta(timeframe.duration)
    if convention is TimestampConvention.OPEN:
        return stamps, stamps + duration
    return stamps - duration, stamps


def floor_to_timeframe(
    stamps: pd.DatetimeIndex, timeframe: Timeframe, origin: pd.Timestamp | None = None
) -> pd.DatetimeIndex:
    """Floor timestamps to the start of their containing bar.

    Args:
        stamps: UTC timestamps.
        timeframe: Target timeframe.
        origin: Grid anchor. Required for timeframes that do not divide the day
            evenly from midnight (e.g. a broker day starting at 22:00 UTC).
    """
    freq = pd.Timedelta(timeframe.duration)
    if origin is None:
        return pd.DatetimeIndex(stamps.floor(freq))
    offsets = ((stamps - origin) // freq) * freq
    return pd.DatetimeIndex(origin + offsets)


def is_on_grid(stamps: pd.DatetimeIndex, timeframe: Timeframe) -> pd.Series:
    """Boolean mask of timestamps that sit exactly on the timeframe grid.

    Off-grid timestamps usually mean the source file mixes timeframes or the
    broker emitted a partial bar; either way it must be surfaced, not silently
    floored away.
    """
    floored = floor_to_timeframe(stamps, timeframe)
    return pd.Series(floored == stamps, index=stamps)


def expected_grid(start: pd.Timestamp, end: pd.Timestamp, timeframe: Timeframe) -> pd.DatetimeIndex:
    """Every timestamp the timeframe grid should contain between two bounds.

    This is the *calendar* grid; it does not know about market closures. Use
    :mod:`xaubot.core.calendar` to filter it down to expected trading bars.
    """
    if end < start:
        raise DataError(f"end ({end}) precedes start ({start})")
    return pd.date_range(start=start, end=end, freq=timeframe.pandas_freq, tz=UTC)


def bars_between(start: pd.Timestamp, end: pd.Timestamp, timeframe: Timeframe) -> int:
    """Number of whole timeframe bars between two timestamps."""
    return int((end - start) // pd.Timedelta(timeframe.duration))


def to_utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    """Coerce a timestamp or ISO string to tz-aware UTC.

    Handles both naive and already-aware input, which the plain
    ``pd.Timestamp(value, tz="UTC")`` constructor does not: it raises on
    aware input rather than converting.
    """
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(UTC) if stamp.tzinfo is None else stamp.tz_convert(UTC)


def to_epoch_ms(stamps: pd.DatetimeIndex) -> pd.Index:
    """Convert to integer milliseconds since epoch (stable Parquet sort key)."""
    return stamps.astype("int64") // 1_000_000
