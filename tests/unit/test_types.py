"""Domain-type invariants.

``BarFrame`` refuses to exist in an invalid state. That is deliberate: the
alternative is every downstream module re-checking (or forgetting to check)
whether its input is sorted, unique, and timezone-aware.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import make_bars
from xaubot.core.enums import Timeframe
from xaubot.core.errors import SchemaError
from xaubot.core.time_utils import CLOSE_TIME, OPEN_TIME
from xaubot.core.types import BarFrame, InstrumentSpec


def _indexed(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.set_index(CLOSE_TIME, drop=False)
    out.index.name = CLOSE_TIME
    return out


class TestBarFrame:
    def test_valid_frame_constructs(self, bar_frame: BarFrame) -> None:
        assert len(bar_frame) > 0
        assert str(bar_frame.close_times.tz) == "UTC"
        assert bar_frame.start < bar_frame.end

    def test_rejects_naive_index(self) -> None:
        frame = _indexed(make_bars(50))
        frame.index = frame.index.tz_localize(None)
        frame[CLOSE_TIME] = frame.index
        with pytest.raises(SchemaError, match="tz-aware"):
            BarFrame(df=frame, timeframe=Timeframe.M5, symbol="XAUUSD")

    def test_rejects_unsorted_index(self) -> None:
        frame = _indexed(make_bars(50)).iloc[::-1]
        with pytest.raises(SchemaError, match="monotonic"):
            BarFrame(df=frame, timeframe=Timeframe.M5, symbol="XAUUSD")

    def test_rejects_duplicate_timestamps(self) -> None:
        frame = _indexed(make_bars(50))
        frame = pd.concat([frame, frame.iloc[[10]]]).sort_index()
        with pytest.raises(SchemaError, match="duplicate"):
            BarFrame(df=frame, timeframe=Timeframe.M5, symbol="XAUUSD")

    def test_rejects_wrong_bar_duration(self) -> None:
        frame = _indexed(make_bars(50))
        frame.loc[frame.index[5], OPEN_TIME] = frame.index[5] - pd.Timedelta(minutes=17)
        with pytest.raises(SchemaError, match="duration"):
            BarFrame(df=frame, timeframe=Timeframe.M5, symbol="XAUUSD")

    def test_rejects_missing_columns(self) -> None:
        frame = _indexed(make_bars(50)).drop(columns=["volume"])
        with pytest.raises(SchemaError, match="missing required columns"):
            BarFrame(df=frame, timeframe=Timeframe.M5, symbol="XAUUSD")

    def test_as_of_is_inclusive_and_excludes_the_future(self, bar_frame: BarFrame) -> None:
        cutoff = bar_frame.close_times[100]
        sliced = bar_frame.as_of(cutoff)
        assert sliced.end == cutoff
        assert len(sliced) == 101
        assert (sliced.close_times <= cutoff).all()

    def test_daily_bars_may_be_23_or_25_hours(self) -> None:
        """DST transitions make an anchored trading day 23h or 25h long."""
        opens = pd.DatetimeIndex(
            [pd.Timestamp("2026-03-06 22:00", tz="UTC"), pd.Timestamp("2026-03-07 22:00", tz="UTC")]
        )
        closes = opens + pd.to_timedelta([24, 23], unit="h")
        frame = pd.DataFrame(
            {
                OPEN_TIME: opens,
                CLOSE_TIME: closes,
                "open": [1.0, 1.0],
                "high": [2.0, 2.0],
                "low": [0.5, 0.5],
                "close": [1.5, 1.5],
                "volume": [10.0, 10.0],
            },
            index=closes,
        )
        frame.index.name = CLOSE_TIME
        BarFrame(df=frame, timeframe=Timeframe.D1, symbol="XAUUSD")  # must not raise


class TestInstrumentSpec:
    spec = InstrumentSpec(symbol="XAUUSD")

    def test_value_per_lot(self) -> None:
        # A $7.40 move on 1.00 lot with 0.01 tick size and $1/tick = $740.
        assert self.spec.value_per_lot(7.40) == pytest.approx(740.0)

    def test_round_lots_floors_to_step(self) -> None:
        assert self.spec.round_lots(0.1352) == pytest.approx(0.13)

    def test_round_lots_below_minimum_is_zero(self) -> None:
        """Rounding a sub-minimum size UP would silently exceed the risk budget."""
        assert self.spec.round_lots(0.004) == 0.0

    def test_round_lots_clamps_to_max(self) -> None:
        assert self.spec.round_lots(999.0) == pytest.approx(self.spec.max_lot)

    def test_rejects_invalid_spec(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            InstrumentSpec(symbol="X", tick_size=0.0)
        with pytest.raises(ValueError, match="min_lot exceeds max_lot"):
            InstrumentSpec(symbol="X", min_lot=10.0, max_lot=1.0)
