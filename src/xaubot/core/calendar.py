"""XAUUSD trading calendar.

Gap detection is meaningless without a model of when the market is *supposed*
to be open. Spot gold trades roughly 23 hours a day, five days a week, with a
daily maintenance break and a weekend closure -- so a naive "every 5 minutes"
grid reports ~25% of the week as missing data and buries the gaps that actually
matter (a broker feed outage during London).

The schedule is anchored to New York local time because that is what brokers
use, which also makes it automatically DST-correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from xaubot.core.enums import Timeframe
from xaubot.core.time_utils import expected_grid

_MINUTES_PER_DAY = 1440


def _hhmm(value: str) -> int:
    hours, _, minutes = value.partition(":")
    return int(hours) * 60 + int(minutes or 0)


@dataclass(frozen=True, slots=True)
class TradingCalendar:
    """Weekly open/close schedule for a 24x5 instrument.

    Attributes:
        tz: Timezone the schedule is expressed in.
        week_open_weekday: Weekday the trading week opens (0=Mon, 6=Sun).
        week_open_time: Local open time on that weekday.
        week_close_weekday: Weekday the trading week closes.
        week_close_time: Local close time on that weekday.
        daily_break_start: Local start of the daily maintenance break.
        daily_break_end: Local end of the daily maintenance break.
        holidays: Local dates on which the market is treated as fully closed.
    """

    tz: str = "America/New_York"
    week_open_weekday: int = 6  # Sunday
    week_open_time: str = "18:00"
    week_close_weekday: int = 4  # Friday
    week_close_time: str = "17:00"
    daily_break_start: str = "17:00"
    daily_break_end: str = "18:00"
    holidays: frozenset[date] = field(default_factory=frozenset)

    def is_open(self, utc_index: pd.DatetimeIndex) -> np.ndarray:
        """Boolean mask of timestamps at which the market is open.

        Args:
            utc_index: Timestamps to test (treated as bar *open* times).

        Returns:
            A boolean numpy array aligned with ``utc_index``.
        """
        local = utc_index.tz_convert(self.tz)
        weekday = np.asarray(local.weekday)
        minutes = np.asarray(local.hour * 60 + local.minute)

        open_m = _hhmm(self.week_open_time)
        close_m = _hhmm(self.week_close_time)
        break_start = _hhmm(self.daily_break_start)
        break_end = _hhmm(self.daily_break_end)

        # Minutes elapsed since the start of the local week (Monday 00:00).
        week_minute = weekday * _MINUTES_PER_DAY + minutes
        open_week_minute = self.week_open_weekday * _MINUTES_PER_DAY + open_m
        close_week_minute = self.week_close_weekday * _MINUTES_PER_DAY + close_m

        # The week wraps: open on Sunday evening, close on Friday afternoon.
        if open_week_minute > close_week_minute:
            is_open = (week_minute >= open_week_minute) | (week_minute < close_week_minute)
        else:
            is_open = (week_minute >= open_week_minute) & (week_minute < close_week_minute)

        # Daily maintenance break, on days that are not the weekly open/close day.
        in_break = (minutes >= break_start) & (minutes < break_end)
        in_break &= weekday != self.week_open_weekday
        in_break &= weekday != self.week_close_weekday
        is_open &= ~in_break

        if self.holidays:
            local_dates = local.date
            holiday_mask = np.fromiter(
                (d in self.holidays for d in local_dates), dtype=bool, count=len(local_dates)
            )
            is_open &= ~holiday_mask

        return np.asarray(is_open)

    def expected_bars(self, start: pd.Timestamp, end: pd.Timestamp, timeframe: Timeframe) -> pd.DatetimeIndex:
        """Bar open times the feed should contain between two bounds.

        Only meaningful for intraday timeframes; daily bars are defined by the
        resampler's day boundary rather than by this grid.
        """
        grid = expected_grid(start, end, timeframe)
        return grid[self.is_open(grid)]

    def trading_day(self, utc_index: pd.DatetimeIndex) -> pd.Index:
        """Broker trading day each timestamp belongs to.

        The trading day rolls at the weekly/daily open time, so the bars from
        18:05 NY Sunday onward belong to Monday's trading day. Daily-loss limits
        and daily P&L must use this, not the UTC calendar date -- otherwise the
        loss counter resets in the middle of the New York session.
        """
        local = utc_index.tz_convert(self.tz)
        roll_minute = _hhmm(self.daily_break_end)
        minutes = local.hour * 60 + local.minute
        offset = np.where(minutes >= roll_minute, 1, 0)
        return pd.Index(local.normalize() + pd.to_timedelta(offset, unit="D")).date


DEFAULT_CALENDAR = TradingCalendar()
