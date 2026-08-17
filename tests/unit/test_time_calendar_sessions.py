"""Timestamp, calendar, and session tests.

The DST cases matter more than they look: London and New York change clocks on
different dates, so a hard-coded UTC session window is wrong for several weeks
a year -- and those weeks include the London/NY overlap, which is where most of
gold's tradable volatility lives.
"""

from __future__ import annotations

import pandas as pd
import pytest

from xaubot.core.calendar import TradingCalendar
from xaubot.core.enums import Session, Timeframe, TimestampConvention
from xaubot.core.errors import DataError
from xaubot.core.sessions import classify_sessions
from xaubot.core.time_utils import (
    bars_between,
    derive_bar_times,
    ensure_utc,
    expected_grid,
    floor_to_timeframe,
    is_on_grid,
)


def utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def idx(*values: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(v, tz="UTC") for v in values])


class TestTimeUtils:
    def test_ensure_utc_localises_naive(self) -> None:
        naive = pd.DatetimeIndex(["2026-03-02 12:00", "2026-03-02 13:00"])
        result = ensure_utc(naive, "Europe/London")
        assert str(result.tz) == "UTC"
        assert result[0] == utc("2026-03-02 12:00")  # GMT in March => no offset

    def test_ensure_utc_converts_aware(self) -> None:
        aware = pd.DatetimeIndex(["2026-07-02 12:00-04:00"])
        assert ensure_utc(aware)[0] == utc("2026-07-02 16:00")

    def test_derive_bar_times_open_convention(self) -> None:
        stamps = idx("2026-01-05 10:10")
        open_time, close_time = derive_bar_times(stamps, Timeframe.M5, TimestampConvention.OPEN)
        assert open_time[0] == utc("2026-01-05 10:10")
        assert close_time[0] == utc("2026-01-05 10:15")

    def test_derive_bar_times_close_convention(self) -> None:
        """The same stamp means a different bar under the other convention.

        This is exactly the one-bar shift that silently creates look-ahead bias.
        """
        stamps = idx("2026-01-05 10:10")
        open_time, close_time = derive_bar_times(stamps, Timeframe.M5, TimestampConvention.CLOSE)
        assert open_time[0] == utc("2026-01-05 10:05")
        assert close_time[0] == utc("2026-01-05 10:10")

    def test_floor_and_grid(self) -> None:
        stamps = idx("2026-01-05 10:07", "2026-01-05 10:10")
        assert list(floor_to_timeframe(stamps, Timeframe.M5)) == [
            utc("2026-01-05 10:05"),
            utc("2026-01-05 10:10"),
        ]
        assert is_on_grid(stamps, Timeframe.M5).tolist() == [False, True]

    def test_floor_with_origin(self) -> None:
        origin = utc("1970-01-01 02:00")
        stamps = idx("2026-01-05 09:00")
        assert floor_to_timeframe(stamps, Timeframe.H4, origin=origin)[0] == utc("2026-01-05 06:00")

    def test_expected_grid_rejects_reversed_bounds(self) -> None:
        with pytest.raises(DataError):
            expected_grid(utc("2026-01-02"), utc("2026-01-01"), Timeframe.M5)

    def test_bars_between(self) -> None:
        assert bars_between(utc("2026-01-05 00:00"), utc("2026-01-05 01:00"), Timeframe.M5) == 12


class TestCalendar:
    calendar = TradingCalendar()

    def test_weekend_is_closed(self) -> None:
        # Saturday, any hour.
        assert not self.calendar.is_open(idx("2026-01-10 12:00"))[0]

    def test_sunday_open_boundary(self) -> None:
        # 2026-01-11 is a Sunday. 18:00 New York = 23:00 UTC in January (EST).
        assert not self.calendar.is_open(idx("2026-01-11 22:30"))[0]
        assert self.calendar.is_open(idx("2026-01-11 23:30"))[0]

    def test_friday_close_boundary(self) -> None:
        # 2026-01-09 is a Friday. 17:00 New York = 22:00 UTC.
        assert self.calendar.is_open(idx("2026-01-09 21:30"))[0]
        assert not self.calendar.is_open(idx("2026-01-09 22:30"))[0]

    def test_daily_maintenance_break(self) -> None:
        # Wednesday 17:00-18:00 New York = 22:00-23:00 UTC in January.
        assert not self.calendar.is_open(idx("2026-01-07 22:30"))[0]
        assert self.calendar.is_open(idx("2026-01-07 23:30"))[0]

    def test_dst_shifts_the_break_in_utc(self) -> None:
        """In July the break is 21:00-22:00 UTC, not 22:00-23:00."""
        assert not self.calendar.is_open(idx("2026-07-08 21:30"))[0]
        assert self.calendar.is_open(idx("2026-07-08 22:30"))[0]

    def test_holidays_close_the_market(self) -> None:
        cal = TradingCalendar(holidays=frozenset({pd.Timestamp("2026-12-25").date()}))
        assert not cal.is_open(idx("2026-12-25 15:00"))[0]

    def test_expected_bars_excludes_weekend(self) -> None:
        bars = self.calendar.expected_bars(utc("2026-01-09 00:00"), utc("2026-01-12 00:00"), Timeframe.H1)
        weekend = [b for b in bars if b.weekday() == 5]
        assert not weekend

    def test_trading_day_rolls_at_the_open(self) -> None:
        """Bars after the 18:00 New York roll belong to the next trading day."""
        days = self.calendar.trading_day(idx("2026-01-12 22:30", "2026-01-12 23:30"))
        assert days[0] == pd.Timestamp("2026-01-12").date()
        assert days[1] == pd.Timestamp("2026-01-13").date()


class TestSessions:
    def test_london_open_tracks_dst(self) -> None:
        """London opens 08:00 UTC in winter but 07:00 UTC in summer."""
        winter = classify_sessions(idx("2026-01-15 07:30", "2026-01-15 08:30"))
        assert not winter["in_london"].iloc[0]
        assert winter["in_london"].iloc[1]

        summer = classify_sessions(idx("2026-07-15 06:30", "2026-07-15 07:30"))
        assert not summer["in_london"].iloc[0]
        assert summer["in_london"].iloc[1]

    def test_overlap_window_moves_with_dst(self) -> None:
        """12:30 UTC is the overlap in July but London-only in January."""
        summer = classify_sessions(idx("2026-07-15 12:30"))
        assert summer["session"].iloc[0] == Session.LONDON_NY_OVERLAP.value

        winter = classify_sessions(idx("2026-01-15 12:30"))
        assert winter["session"].iloc[0] == Session.LONDON.value

    def test_asia_session(self) -> None:
        # Tokyo 09:00-18:00 = 00:00-09:00 UTC year round (Japan has no DST).
        result = classify_sessions(idx("2026-01-15 02:00"))
        assert result["session"].iloc[0] == Session.ASIA.value

    def test_off_session(self) -> None:
        result = classify_sessions(idx("2026-01-15 23:00"))
        assert result["session"].iloc[0] == Session.OFF.value

    def test_session_id_groups_one_session_instance(self) -> None:
        stamps = idx("2026-01-15 09:00", "2026-01-15 10:00", "2026-01-16 09:00")
        result = classify_sessions(stamps)
        assert result["session_id"].iloc[0] == result["session_id"].iloc[1]
        assert result["session_id"].iloc[0] != result["session_id"].iloc[2]

    def test_weekend_is_not_a_session(self) -> None:
        result = classify_sessions(idx("2026-01-10 12:00"))
        assert result["session"].iloc[0] == Session.OFF.value
