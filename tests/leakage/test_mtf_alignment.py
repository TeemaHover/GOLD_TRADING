"""Point-in-time guards for multi-timeframe alignment.

These are the most important tests in the repository. Every one of them
corresponds to a row in the leakage audit register (docs/ARCHITECTURE.md
section 12). A failure here invalidates every downstream result, so they are
written to fail loudly rather than to be convenient.

The canonical failure they exist to prevent: a 5m decision at 10:15 reading the
15m bar that covers 10:15-10:30. That bar does not exist yet at 10:15, but it
is the row a naive `reindex(method="ffill")` or a merge on open time will hand
you -- and a model given it will look extraordinarily accurate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars
from xaubot.core.enums import Timeframe
from xaubot.core.time_utils import CLOSE_TIME, OPEN_TIME
from xaubot.core.types import BarFrame
from xaubot.data.resample import resample_bars

pytestmark = pytest.mark.leakage

CONTEXT_TIMEFRAMES = (Timeframe.M15, Timeframe.H1, Timeframe.H4)


@pytest.fixture(scope="module")
def base() -> BarFrame:
    frame = make_bars(5000, start="2026-01-05 00:00").set_index(CLOSE_TIME, drop=False)
    frame.index.name = CLOSE_TIME
    return BarFrame(df=frame, timeframe=Timeframe.M5, symbol="XAUUSD")


def as_of_merge(base: BarFrame, htf: BarFrame) -> pd.DataFrame:
    """The exact merge the feature engine will use for HTF context."""
    left = base.df[[CLOSE_TIME]].reset_index(drop=True).sort_values(CLOSE_TIME)
    right = (
        htf.df[[CLOSE_TIME, OPEN_TIME, "high", "low", "close"]]
        .reset_index(drop=True)
        .sort_values(CLOSE_TIME)
        .rename(columns={CLOSE_TIME: "htf_close_time"})
    )
    return pd.merge_asof(
        left,
        right,
        left_on=CLOSE_TIME,
        right_on="htf_close_time",
        direction="backward",
        allow_exact_matches=True,
    )


class TestAggregationIsCausal:
    @pytest.mark.parametrize("timeframe", CONTEXT_TIMEFRAMES)
    def test_htf_bar_never_aggregates_a_bar_that_closes_after_it(
        self, base: BarFrame, timeframe: Timeframe
    ) -> None:
        """Every constituent 5m bar must close at or before the HTF close."""
        htf = resample_bars(base, timeframe)
        base_df = base.df

        for row in htf.df.itertuples():
            members = base_df[
                (base_df[OPEN_TIME] >= getattr(row, OPEN_TIME)) & (base_df[OPEN_TIME] < row.close_time)
            ]
            assert len(members) > 0
            assert (members[CLOSE_TIME] <= row.close_time).all()
            # And the aggregate really is the aggregate of exactly those bars.
            assert row.high == pytest.approx(members["high"].max())
            assert row.low == pytest.approx(members["low"].min())
            assert row.close == pytest.approx(members["close"].iloc[-1])


class TestAsOfMergeIsCausal:
    @pytest.mark.parametrize("timeframe", CONTEXT_TIMEFRAMES)
    def test_selected_htf_bar_has_already_closed(self, base: BarFrame, timeframe: Timeframe) -> None:
        merged = as_of_merge(base, resample_bars(base, timeframe))
        matched = merged.dropna(subset=["htf_close_time"])
        assert len(matched) > 0
        violations = matched[matched["htf_close_time"] > matched[CLOSE_TIME]]
        assert violations.empty, (
            f"{len(violations)} rows read an unfinished {timeframe.value} bar; "
            f"first at {violations[CLOSE_TIME].iloc[0] if len(violations) else None}"
        )

    @pytest.mark.parametrize("timeframe", CONTEXT_TIMEFRAMES)
    def test_context_is_never_staler_than_one_htf_bar(self, base: BarFrame, timeframe: Timeframe) -> None:
        """Guards the opposite error: matching too far back and losing context.

        Only bars with a fully contiguous lookback are checked. After a weekend
        or the daily maintenance break the newest closed HTF bar legitimately
        predates the current bar by more than one HTF period, and asserting
        otherwise would be testing the market calendar, not the merge.
        """
        merged = as_of_merge(base, resample_bars(base, timeframe)).dropna(subset=["htf_close_time"])
        staleness = merged[CLOSE_TIME] - merged["htf_close_time"]

        window = timeframe.minutes // Timeframe.M5.minutes
        contiguous = merged[CLOSE_TIME].diff().eq(pd.Timedelta(minutes=5))
        fully_contiguous = contiguous.rolling(window).min().fillna(0).astype(bool)

        assert fully_contiguous.sum() > 100, "not enough contiguous bars to test"
        assert staleness[fully_contiguous].max() < pd.Timedelta(timeframe.duration)

    def test_the_1015_worked_example(self) -> None:
        """The concrete case from docs/ARCHITECTURE.md section 1.4.

        At 10:15 the 15m bar covering 10:00-10:15 is available (it closes
        exactly at 10:15). The bar covering 10:15-10:30 is not.
        """
        opens = pd.date_range(pd.Timestamp("2026-01-06 10:00", tz="UTC"), periods=6, freq="5min")
        frame = pd.DataFrame(
            {
                OPEN_TIME: opens,
                CLOSE_TIME: opens + pd.Timedelta(minutes=5),
                "open": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "high": [1.0, 2.0, 3.0, 400.0, 5.0, 6.0],  # spike lives in the FUTURE bar
                "low": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "volume": [1.0] * 6,
            },
            index=opens + pd.Timedelta(minutes=5),
        )
        frame.index.name = CLOSE_TIME
        base = BarFrame(df=frame, timeframe=Timeframe.M5, symbol="XAUUSD")

        htf = resample_bars(base, Timeframe.M15)
        merged = as_of_merge(base, htf).set_index(CLOSE_TIME)

        at_1015 = merged.loc[pd.Timestamp("2026-01-06 10:15", tz="UTC")]
        assert at_1015["htf_close_time"] == pd.Timestamp("2026-01-06 10:15", tz="UTC")
        # The 400.0 spike is in the 10:15-10:30 bar. If it leaks, this fails.
        assert at_1015["high"] == pytest.approx(3.0)

        at_1020 = merged.loc[pd.Timestamp("2026-01-06 10:20", tz="UTC")]
        assert at_1020["htf_close_time"] == pd.Timestamp("2026-01-06 10:15", tz="UTC")
        assert at_1020["high"] == pytest.approx(3.0)

    def test_naive_forward_fill_on_open_time_does_leak(self) -> None:
        """Demonstrates the bug these tests exist to catch.

        Kept as a test so the failure mode stays documented and anyone tempted
        to "simplify" the merge can see what they would be reintroducing.
        """
        opens = pd.date_range(pd.Timestamp("2026-01-06 10:00", tz="UTC"), periods=6, freq="5min")
        htf_open = pd.Timestamp("2026-01-06 10:15", tz="UTC")
        htf = pd.DataFrame(
            {"htf_high": [3.0, 400.0]},
            index=[pd.Timestamp("2026-01-06 10:00", tz="UTC"), htf_open],
        )

        leaked = htf.reindex(opens, method="ffill")
        # At 10:15 open time, the naive join already exposes the 10:15-10:30 high.
        assert leaked.loc[htf_open, "htf_high"] == pytest.approx(400.0)


class TestReplayEquivalence:
    """Truncating history must not change already-closed HTF bars.

    This is the precondition for backtest/live parity: if resampling the first
    N bars gives different context than resampling the first N+k and slicing,
    then the backtest is seeing something the live system never will.
    """

    @pytest.mark.parametrize("timeframe", CONTEXT_TIMEFRAMES)
    def test_truncated_history_is_a_prefix_of_full_history(
        self, base: BarFrame, timeframe: Timeframe
    ) -> None:
        full = resample_bars(base, timeframe)
        rng = np.random.default_rng(11)

        for cutoff_pos in rng.integers(500, len(base) - 1, size=8):
            cutoff = base.close_times[int(cutoff_pos)]
            truncated = resample_bars(base.as_of(cutoff), timeframe)

            expected = full.df.loc[full.df[CLOSE_TIME] <= cutoff]
            common = expected.index.intersection(truncated.df.index)
            assert len(common) > 0

            for column in ("open", "high", "low", "close", "volume"):
                np.testing.assert_allclose(
                    truncated.df.loc[common, column].to_numpy(),
                    expected.loc[common, column].to_numpy(),
                    rtol=1e-12,
                    err_msg=f"{timeframe.value} {column} changed when history was truncated at {cutoff}",
                )

            # Nothing that closes after the cutoff may appear at all.
            assert (truncated.df[CLOSE_TIME] <= cutoff).all()
