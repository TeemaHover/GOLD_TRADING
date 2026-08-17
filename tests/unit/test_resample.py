"""Higher-timeframe construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars
from xaubot.config.schema import ResampleConfig
from xaubot.core.calendar import DEFAULT_CALENDAR
from xaubot.core.enums import Timeframe
from xaubot.core.errors import ResampleError
from xaubot.core.time_utils import CLOSE_TIME, OPEN_TIME
from xaubot.core.types import BarFrame
from xaubot.data.resample import resample_bars


def hand_built(n_bars: int = 12, start: str = "2026-01-06 10:00") -> BarFrame:
    """A small 5m frame with known values, inside trading hours."""
    opens = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n_bars, freq="5min")
    frame = pd.DataFrame(
        {
            OPEN_TIME: opens,
            CLOSE_TIME: opens + pd.Timedelta(minutes=5),
            "open": np.arange(n_bars, dtype="float64") + 4000.0,
            "high": np.arange(n_bars, dtype="float64") + 4001.0,
            "low": np.arange(n_bars, dtype="float64") + 3999.0,
            "close": np.arange(n_bars, dtype="float64") + 4000.5,
            "volume": np.full(n_bars, 100.0),
        },
        index=opens + pd.Timedelta(minutes=5),
    )
    frame.index.name = CLOSE_TIME
    return BarFrame(df=frame, timeframe=Timeframe.M5, symbol="XAUUSD")


class TestResampleMechanics:
    def test_ohlcv_aggregation_is_correct(self) -> None:
        result = resample_bars(hand_built(12), Timeframe.M15)
        assert len(result) == 4

        first = result.df.iloc[0]
        assert first["open"] == pytest.approx(4000.0)  # first open of the group
        assert first["high"] == pytest.approx(4003.0)  # max over bars 0-2
        assert first["low"] == pytest.approx(3999.0)  # min over bars 0-2
        assert first["close"] == pytest.approx(4002.5)  # last close of the group
        assert first["volume"] == pytest.approx(300.0)  # summed
        assert first["n_base_bars"] == 3

    def test_bars_are_stamped_with_their_close_time(self) -> None:
        result = resample_bars(hand_built(12), Timeframe.M15)
        assert result.df[OPEN_TIME].iloc[0] == pd.Timestamp("2026-01-06 10:00", tz="UTC")
        assert result.df[CLOSE_TIME].iloc[0] == pd.Timestamp("2026-01-06 10:15", tz="UTC")
        assert (result.df[CLOSE_TIME] - result.df[OPEN_TIME] == pd.Timedelta(minutes=15)).all()

    def test_index_equals_close_time(self) -> None:
        result = resample_bars(hand_built(12), Timeframe.M15)
        assert result.df.index.equals(pd.DatetimeIndex(result.df[CLOSE_TIME]))

    def test_trailing_incomplete_bar_is_dropped(self) -> None:
        """11 bars = three complete 15m bars plus a partial one, which must go."""
        result = resample_bars(hand_built(11), Timeframe.M15)
        assert len(result) == 3
        assert result.end == pd.Timestamp("2026-01-06 10:45", tz="UTC")

    def test_partial_bars_are_flagged_not_deleted(self) -> None:
        """Silently deleting a holiday half-day would be data destruction."""
        bars = hand_built(12)
        trimmed = bars.df.drop(index=bars.df.index[3:5])
        result = resample_bars(BarFrame(df=trimmed, timeframe=Timeframe.M5, symbol="XAUUSD"), Timeframe.M15)
        assert result.df["is_partial"].any()
        assert len(result) >= 3

    def test_upsampling_is_refused(self) -> None:
        with pytest.raises(ResampleError, match="upsampling"):
            resample_bars(hand_built(12), Timeframe.M5)

    def test_all_context_timeframes_build(self) -> None:
        base = BarFrame(
            df=make_bars(4000).set_index(CLOSE_TIME, drop=False),
            timeframe=Timeframe.M5,
            symbol="XAUUSD",
        )
        for tf in (Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1):
            result = resample_bars(base, tf, ResampleConfig(), DEFAULT_CALENDAR)
            assert len(result) > 0
            assert result.df.index.is_monotonic_increasing

    def test_conservation_of_extremes(self) -> None:
        """No aggregation may invent a high above (or low below) the source."""
        base = BarFrame(
            df=make_bars(4000).set_index(CLOSE_TIME, drop=False),
            timeframe=Timeframe.M5,
            symbol="XAUUSD",
        )
        result = resample_bars(base, Timeframe.H1)
        assert result.df["high"].max() <= base.df["high"].max()
        assert result.df["low"].min() >= base.df["low"].min()


class TestDailyAnchoring:
    def test_daily_bar_closes_at_the_new_york_close(self) -> None:
        base = BarFrame(
            df=make_bars(6000, start="2026-01-05 00:00").set_index(CLOSE_TIME, drop=False),
            timeframe=Timeframe.M5,
            symbol="XAUUSD",
        )
        daily = resample_bars(base, Timeframe.D1, ResampleConfig())
        local_close = daily.df[CLOSE_TIME].dt.tz_convert("America/New_York")
        assert (local_close.dt.hour == 17).all()
        assert (local_close.dt.minute == 0).all()

    def test_daily_anchor_is_configurable(self) -> None:
        base = BarFrame(
            df=make_bars(6000, start="2026-01-05 00:00").set_index(CLOSE_TIME, drop=False),
            timeframe=Timeframe.M5,
            symbol="XAUUSD",
        )
        config = ResampleConfig(daily_anchor_tz="UTC", daily_anchor_time="00:00")
        daily = resample_bars(base, Timeframe.D1, config)
        assert (daily.df[CLOSE_TIME].dt.hour == 0).all()

    def test_h4_anchor_offset(self) -> None:
        base = BarFrame(
            df=make_bars(2000, start="2026-01-05 00:00").set_index(CLOSE_TIME, drop=False),
            timeframe=Timeframe.M5,
            symbol="XAUUSD",
        )
        aligned = resample_bars(base, Timeframe.H4, ResampleConfig())
        offset = resample_bars(base, Timeframe.H4, ResampleConfig(h4_anchor_offset_minutes=120))
        assert set(aligned.df[OPEN_TIME].dt.hour) <= {0, 4, 8, 12, 16, 20}
        assert set(offset.df[OPEN_TIME].dt.hour) <= {2, 6, 10, 14, 18, 22}
