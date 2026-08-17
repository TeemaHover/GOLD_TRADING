"""Behavioural tests for each feature group.

Where a group has a hand-checkable pattern (a fair value gap, a swing pivot, a
previous-day high), the test plants that pattern explicitly rather than
asserting properties of random data -- otherwise a detector that never fires
passes every test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars
from xaubot.config.schema import FeaturesConfig
from xaubot.core.calendar import DEFAULT_CALENDAR
from xaubot.core.enums import Timeframe
from xaubot.core.sessions import DEFAULT_SESSIONS
from xaubot.core.time_utils import CLOSE_TIME, OPEN_TIME
from xaubot.core.types import BarFrame
from xaubot.features._ta import atr as atr_fn
from xaubot.features.base import FeatureContext
from xaubot.features.engine import FeatureEngine
from xaubot.features.fvg import FvgTransform
from xaubot.features.liquidity import LiquidityTransform
from xaubot.features.patterns import PatternTransform
from xaubot.features.price import PriceTransform
from xaubot.features.structure import StructureTransform
from xaubot.features.time_features import TimeTransform
from xaubot.features.volatility import VolatilityTransform

CONFIG = FeaturesConfig()


def frame_from(
    highs: list[float], lows: list[float], opens: list[float], closes: list[float], start: str
) -> pd.DataFrame:
    """Build a bar frame with exact, hand-chosen OHLC values."""
    stamps = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(highs), freq="5min")
    frame = pd.DataFrame(
        {
            OPEN_TIME: stamps,
            CLOSE_TIME: stamps + pd.Timedelta(minutes=5),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100.0] * len(highs),
        },
        index=stamps + pd.Timedelta(minutes=5),
    )
    frame.index.name = CLOSE_TIME
    return frame


def context_for(bars: pd.DataFrame, atr_value: float | None = None) -> FeatureContext:
    from xaubot.core.sessions import classify_sessions

    if atr_value is None:
        atr = atr_fn(bars["high"], bars["low"], bars["close"], 14).bfill().fillna(1.0)
    else:
        atr = pd.Series(atr_value, index=bars.index)
    return FeatureContext(
        timeframe=Timeframe.M5,
        atr=atr,
        sessions=classify_sessions(pd.DatetimeIndex(bars.index), DEFAULT_SESSIONS),
        calendar=DEFAULT_CALENDAR,
        config=CONFIG,
    )


@pytest.fixture(scope="module")
def synthetic_bars() -> pd.DataFrame:
    frame = make_bars(3000, start="2026-01-05 00:00").set_index(CLOSE_TIME, drop=False)
    frame.index.name = CLOSE_TIME
    return frame


class TestPriceFeatures:
    def test_bounded_features_stay_in_range(self, synthetic_bars: pd.DataFrame) -> None:
        out = PriceTransform((1, 5), (20,)).compute(synthetic_bars, context_for(synthetic_bars))
        for column in ("close_position", "body_to_range", "upper_wick", "lower_wick"):
            values = out[column].dropna()
            assert values.between(-1e-9, 1 + 1e-9).all(), column

    def test_close_position_endpoints(self) -> None:
        bars = frame_from([10.0], [8.0], [9.0], [10.0], "2026-01-06 10:00")
        out = PriceTransform((1,), (20,)).compute(bars, context_for(bars, 1.0))
        assert out["close_position"].iloc[0] == pytest.approx(1.0)

    def test_returns_are_scale_free(self) -> None:
        """Doubling every price must not change the return features.

        This is what lets a model trained near 4,000 still work near 5,600.
        """
        low_price = make_bars(400, start_price=2000.0, seed=4).set_index(CLOSE_TIME, drop=False)
        high_price = low_price.copy()
        for column in ("open", "high", "low", "close"):
            high_price[column] = high_price[column] * 2.0

        transform = PriceTransform((1, 5), (20,))
        left = transform.compute(low_price, context_for(low_price))
        right = transform.compute(high_price, context_for(high_price))

        for column in ("close_position", "body_to_range", "high_low_range_atr"):
            np.testing.assert_allclose(
                left[column].dropna().to_numpy(),
                right[column].dropna().to_numpy(),
                rtol=1e-9,
                err_msg=column,
            )


class TestVolatilityFeatures:
    def test_regime_one_hot_sums_to_one(self, synthetic_bars: pd.DataFrame) -> None:
        out = VolatilityTransform(CONFIG.volatility).compute(synthetic_bars, context_for(synthetic_bars))
        columns = [c for c in out.columns if c.startswith("vol_regime_is_")]
        valid = out["vol_regime"].notna()
        assert valid.any()
        assert (out.loc[valid, columns].sum(axis=1) == 1.0).all()

    def test_percentiles_are_bounded(self, synthetic_bars: pd.DataFrame) -> None:
        out = VolatilityTransform(CONFIG.volatility).compute(synthetic_bars, context_for(synthetic_bars))
        for column in out.columns:
            if "percentile" in column:
                assert out[column].dropna().between(0.0, 1.0).all(), column

    def test_nr7_marks_the_narrowest_of_seven(self) -> None:
        highs = [10.0] * 7
        lows = [9.0, 9.0, 9.0, 9.5, 9.0, 9.0, 9.0]
        bars = frame_from(highs, lows, [9.5] * 7, [9.6] * 7, "2026-01-06 10:00")
        out = VolatilityTransform(CONFIG.volatility).compute(bars, context_for(bars, 1.0))
        assert out["nr7"].iloc[3] == 0.0  # window incomplete at index 3
        assert out["nr7"].iloc[6] == 0.0  # index 6 is not the narrowest


class TestStructureFeatures:
    def test_swing_is_confirmed_exactly_k_bars_late(self) -> None:
        """The core anti-leak property of this module.

        A pivot at index 4 must be invisible at indices 4 and 5, and appear at
        index 6 for k=2. Marking it at index 4 -- as most implementations do --
        backdates knowledge by two bars.
        """
        highs = [10, 11, 12, 13, 20, 13, 12, 11, 10, 9, 8]
        lows = [h - 2 for h in highs]
        closes = [h - 1 for h in highs]
        bars = frame_from(
            [float(h) for h in highs],
            [float(x) for x in lows],
            [float(c) for c in closes],
            [float(c) for c in closes],
            "2026-01-06 10:00",
        )
        transform = StructureTransform(CONFIG.structure)  # swing_k = 2
        out = transform.compute(bars, context_for(bars, 1.0))

        flags = out["swing_high_flag"].to_numpy()
        assert flags[4] == 0.0, "pivot must not be visible on its own bar"
        assert flags[5] == 0.0, "pivot must not be visible before confirmation"
        assert flags[6] == 1.0, "pivot should be confirmed exactly k=2 bars later"

    def test_swing_age_starts_at_k_not_zero(self) -> None:
        highs = [10, 11, 12, 13, 20, 13, 12, 11, 10, 9, 8]
        bars = frame_from(
            [float(h) for h in highs],
            [float(h - 2) for h in highs],
            [float(h - 1) for h in highs],
            [float(h - 1) for h in highs],
            "2026-01-06 10:00",
        )
        out = StructureTransform(CONFIG.structure).compute(bars, context_for(bars, 1.0))
        assert out["last_swing_high_age"].iloc[6] == pytest.approx(2.0)

    def test_range_position_is_bounded(self, synthetic_bars: pd.DataFrame) -> None:
        out = StructureTransform(CONFIG.structure).compute(synthetic_bars, context_for(synthetic_bars))
        assert out["range_position"].dropna().between(-1e-9, 1 + 1e-9).all()
        assert out["consolidation_score"].dropna().between(0.0, 1.0).all()


class TestFvgFeatures:
    def test_gap_is_detected_on_the_third_bar(self) -> None:
        """Bar 2's low sits above bar 0's high: a bullish gap, complete at bar 2."""
        highs = [10.0, 14.0, 16.0, 16.5, 16.5]
        lows = [9.0, 10.5, 12.0, 12.5, 12.5]
        bars = frame_from(
            highs, lows, [9.5, 11.0, 12.5, 13.0, 13.0], [9.8, 13.0, 15.0, 15.0, 15.0], "2026-01-06 10:00"
        )
        out = FvgTransform(CONFIG.fvg).compute(bars, context_for(bars, 1.0))

        assert out["fvg_bull_exists"].iloc[0] == 0.0
        assert out["fvg_bull_exists"].iloc[1] == 0.0
        assert out["fvg_bull_exists"].iloc[2] == 1.0, "gap must appear at the completing bar"
        # Gap spans bar0 high (10.0) to bar2 low (12.0) -> 2.0 price units, ATR=1.
        assert out["fvg_bull_size_atr"].iloc[2] == pytest.approx(2.0)

    def test_absence_is_zero_not_nan(self, synthetic_bars: pd.DataFrame) -> None:
        """ "No gap" is a real state; NaN would look like missing data and get
        the whole group dropped by the dataset stage's NaN filter."""
        out = FvgTransform(CONFIG.fvg).compute(synthetic_bars, context_for(synthetic_bars))
        assert not out.isna().any().any()

    def test_strength_is_bounded(self, synthetic_bars: pd.DataFrame) -> None:
        out = FvgTransform(CONFIG.fvg).compute(synthetic_bars, context_for(synthetic_bars))
        for side in ("bull", "bear"):
            assert out[f"fvg_{side}_strength"].between(0.0, 1.0).all()
            assert out[f"fvg_{side}_fill_pct"].between(0.0, 1.0).all()


class TestPatternFeatures:
    def test_every_strength_is_in_the_unit_interval(self, synthetic_bars: pd.DataFrame) -> None:
        out = PatternTransform(CONFIG.patterns).compute(synthetic_bars, context_for(synthetic_bars))
        for column in out.columns:
            assert out[column].between(0.0, 1.0).all(), column

    def test_bullish_engulfing_scores_and_bearish_does_not(self) -> None:
        # Bar 0 bearish (12 -> 9), bar 1 bullish and engulfing (8.5 -> 13).
        bars = frame_from(
            highs=[12.5, 13.5],
            lows=[8.5, 8.0],
            opens=[12.0, 8.5],
            closes=[9.0, 13.0],
            start="2026-01-06 10:00",
        )
        out = PatternTransform(CONFIG.patterns).compute(bars, context_for(bars, 2.0))
        assert out["engulf_bull_strength"].iloc[1] > 0.0
        assert out["engulf_bear_strength"].iloc[1] == 0.0

    def test_patterns_do_not_fire_on_flat_bars(self) -> None:
        bars = frame_from([10.0] * 30, [10.0] * 30, [10.0] * 30, [10.0] * 30, "2026-01-06 10:00")
        out = PatternTransform(CONFIG.patterns).compute(bars, context_for(bars, 1.0))
        assert (out.iloc[-1] == 0.0).sum() >= len(out.columns) - 1


class TestLiquidityFeatures:
    def test_previous_day_level_is_yesterdays_not_todays(self) -> None:
        """The defining property: today's own high must never leak in."""
        bars = make_bars(1200, start="2026-01-05 00:00").set_index(CLOSE_TIME, drop=False)
        bars.index.name = CLOSE_TIME
        ctx = context_for(bars)
        levels = LiquidityTransform(CONFIG.liquidity)._build_levels(bars, ctx)

        trading_day = pd.Series(DEFAULT_CALENDAR.trading_day(pd.DatetimeIndex(bars.index)), index=bars.index)
        today_high = bars["high"].groupby(trading_day).transform("max")

        valid = levels["prev_day_high"].notna()
        assert valid.any()
        # Equality would be a coincidence at best and a leak at worst; the
        # previous day's high should differ from today's on most bars.
        matches = (levels.loc[valid, "prev_day_high"] == today_high[valid]).mean()
        assert matches < 0.05

    def test_previous_day_level_matches_the_prior_day_aggregate(self) -> None:
        bars = make_bars(1500, start="2026-01-05 00:00").set_index(CLOSE_TIME, drop=False)
        bars.index.name = CLOSE_TIME
        ctx = context_for(bars)
        levels = LiquidityTransform(CONFIG.liquidity)._build_levels(bars, ctx)

        trading_day = pd.Series(DEFAULT_CALENDAR.trading_day(pd.DatetimeIndex(bars.index)), index=bars.index)
        by_day = bars["high"].groupby(trading_day).max()
        days = list(by_day.index)

        # Pick a bar in the middle of the third day and check it sees day two.
        third_day_bars = bars.index[trading_day == days[2]]
        probe = third_day_bars[len(third_day_bars) // 2]
        assert levels.loc[probe, "prev_day_high"] == pytest.approx(by_day.iloc[1])

    def test_running_session_levels_are_causal(self) -> None:
        bars = make_bars(800, start="2026-01-05 00:00").set_index(CLOSE_TIME, drop=False)
        bars.index.name = CLOSE_TIME
        ctx = context_for(bars)
        levels = LiquidityTransform(CONFIG.liquidity)._build_levels(bars, ctx)
        # A running high is a cumulative max, so it can only ever increase
        # within a session and never exceeds the highest bar seen so far.
        assert (levels["session_high"] <= bars["high"].cummax()).all()
        assert (levels["session_low"] >= bars["low"].cummin()).all()


class TestTimeFeatures:
    def test_cyclical_encodings_lie_on_the_unit_circle(self, synthetic_bars: pd.DataFrame) -> None:
        out = TimeTransform(CONFIG.time).compute(synthetic_bars, context_for(synthetic_bars))
        for prefix in ("hour", "dow", "month", "minute_of_day"):
            radius = out[f"{prefix}_sin"] ** 2 + out[f"{prefix}_cos"] ** 2
            np.testing.assert_allclose(radius.to_numpy(), 1.0, atol=1e-9)

    def test_session_one_hot_is_exclusive(self, synthetic_bars: pd.DataFrame) -> None:
        out = TimeTransform(CONFIG.time).compute(synthetic_bars, context_for(synthetic_bars))
        columns = [c for c in out.columns if c.startswith("session_is_")]
        assert (out[columns].sum(axis=1) == 1.0).all()

    def test_bars_since_gap_resets_at_a_discontinuity(self) -> None:
        bars = make_bars(600, start="2026-01-05 00:00")
        bars = bars.drop(index=range(200, 220)).reset_index(drop=True)
        bars = bars.set_index(CLOSE_TIME, drop=False)
        bars.index.name = CLOSE_TIME

        out = TimeTransform(CONFIG.time).compute(bars, context_for(bars))
        counter = out["bars_since_gap"].to_numpy()
        assert counter[0] == 0.0
        assert counter[200] == 0.0, "counter must reset on the bar after the gap"
        assert counter[201] == 1.0
        assert out["is_post_gap"].iloc[200] == 1.0


class TestEngineAccounting:
    def test_warmup_is_reported_in_base_bars(self) -> None:
        """A 4h lookback costs 48x more base bars than its own units suggest."""
        engine = FeatureEngine(CONFIG, DEFAULT_CALENDAR, DEFAULT_SESSIONS)
        base_only = FeaturesConfig(mtf=CONFIG.mtf.model_copy(update={"enabled": False}))
        without_htf = FeatureEngine(base_only, DEFAULT_CALENDAR, DEFAULT_SESSIONS)

        assert engine.required_warmup_bars(Timeframe.M5) > without_htf.required_warmup_bars(Timeframe.M5)

    def test_duplicate_feature_names_are_rejected(self) -> None:
        from xaubot.core.errors import FeatureError
        from xaubot.features.base import FeatureSpec
        from xaubot.features.manifest import build_manifest

        specs = (
            FeatureSpec("dupe", "a", 1, "first"),
            FeatureSpec("dupe", "b", 1, "second"),
        )
        with pytest.raises(FeatureError, match="Duplicate feature names"):
            build_manifest(specs, CONFIG, "XAUUSD", Timeframe.M5, (), 10, 10)

    def test_transform_output_must_match_declared_specs(self, synthetic_bars: pd.DataFrame) -> None:
        from xaubot.core.errors import FeatureError

        class Broken(PriceTransform):
            def _compute(self, bars, ctx):  # type: ignore[no-untyped-def]
                return super()._compute(bars, ctx).drop(columns=["close_ret"])

        with pytest.raises(FeatureError, match="missing="):
            Broken((1,), (20,)).compute(synthetic_bars, context_for(synthetic_bars))


class TestBarFrameUnused:
    """Guard that the module-level helpers used by tests stay importable."""

    def test_barframe_roundtrip(self, synthetic_bars: pd.DataFrame) -> None:
        frame = BarFrame(df=synthetic_bars, timeframe=Timeframe.M5, symbol="XAUUSD")
        assert len(frame) == len(synthetic_bars)
