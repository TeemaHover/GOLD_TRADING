"""Known-answer tests for the indicator primitives.

These exist because the system deliberately does not use a third-party TA
library (several centre windows or backfill by default, which is exactly the
silent look-ahead this project is built to avoid). Rolling your own means
owning the correctness proof, so every primitive gets a case whose answer can
be worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xaubot.features._ta import (
    atr,
    bollinger_width,
    buy_pressure,
    directional_index,
    efficiency_ratio,
    ema,
    hurst_variance_ratio,
    rolling_linreg,
    rolling_percentile,
    rolling_zscore,
    rsi,
    sma,
    squash,
    true_range,
    wilder,
)


def series(values: list[float]) -> pd.Series:
    index = pd.date_range("2026-01-05", periods=len(values), freq="5min", tz="UTC")
    return pd.Series(values, index=index, dtype="float64")


class TestMovingAverages:
    def test_sma_known_answer(self) -> None:
        result = sma(series([1, 2, 3, 4, 5]), 3)
        assert pd.isna(result.iloc[1])  # not enough history
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_ema_known_answer(self) -> None:
        # span=3 -> alpha = 2/(3+1) = 0.5. Seeded at the first valid point.
        result = ema(series([1, 1, 1, 5]), 3)
        assert result.iloc[2] == pytest.approx(1.0)
        assert result.iloc[3] == pytest.approx(1.0 + 0.5 * (5 - 1.0))

    def test_wilder_uses_one_over_n(self) -> None:
        result = wilder(series([2, 2, 2, 6]), 3)
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[3] == pytest.approx(2.0 + (1 / 3) * (6 - 2.0))

    def test_moving_averages_never_look_forward(self) -> None:
        """Appending a huge future value must not change past values."""
        base = series([1, 2, 3, 4, 5])
        extended = series([1, 2, 3, 4, 5, 1000])
        pd.testing.assert_series_equal(sma(base, 3), sma(extended, 3).iloc[:5])
        pd.testing.assert_series_equal(ema(base, 3), ema(extended, 3).iloc[:5])


class TestVolatility:
    def test_true_range_takes_the_widest_of_three(self) -> None:
        high = series([10, 12])
        low = series([9, 11])
        close = series([9.5, 11.5])
        # Bar 1: h-l = 1, |h - prev_close| = 2.5, |l - prev_close| = 1.5 -> 2.5
        assert true_range(high, low, close).iloc[1] == pytest.approx(2.5)

    def test_atr_is_positive_and_finite(self) -> None:
        rng = np.random.default_rng(3)
        close = series(list(4000 + np.cumsum(rng.normal(0, 2, 200))))
        high, low = close + 1.5, close - 1.5
        result = atr(high, low, close, 14).dropna()
        assert (result > 0).all()
        assert np.isfinite(result).all()

    def test_bollinger_width_is_zero_for_a_flat_series(self) -> None:
        assert bollinger_width(series([5.0] * 30), 20).iloc[-1] == pytest.approx(0.0)


class TestMomentum:
    def test_rsi_saturates_high_on_a_pure_uptrend(self) -> None:
        result = rsi(series(list(np.arange(1, 60, dtype="float64"))), 14)
        assert result.iloc[-1] == pytest.approx(100.0, abs=1e-6)

    def test_rsi_saturates_low_on_a_pure_downtrend(self) -> None:
        result = rsi(series(list(np.arange(60, 1, -1, dtype="float64"))), 14)
        assert result.iloc[-1] == pytest.approx(0.0, abs=1e-6)

    def test_efficiency_ratio_is_one_for_a_straight_line(self) -> None:
        """A monotonic move has net displacement equal to its path length."""
        result = efficiency_ratio(series(list(np.arange(1, 40, dtype="float64"))), 20)
        assert result.iloc[-1] == pytest.approx(1.0)

    def test_efficiency_ratio_is_near_zero_for_pure_chop(self) -> None:
        result = efficiency_ratio(series([1.0, 2.0] * 20), 20)
        assert result.iloc[-1] < 0.1

    def test_adx_rises_in_a_trend(self) -> None:
        values = list(np.arange(1, 120, dtype="float64"))
        close = series(values)
        adx, di_plus, di_minus = directional_index(close + 0.5, close - 0.5, close, 14)
        assert di_plus.iloc[-1] > di_minus.iloc[-1]
        assert adx.iloc[-1] > 40


class TestRegression:
    def test_linreg_recovers_a_known_slope(self) -> None:
        slope, r2 = rolling_linreg(series([3.0 * i + 7.0 for i in range(40)]), 20)
        assert slope.iloc[-1] == pytest.approx(3.0)
        assert r2.iloc[-1] == pytest.approx(1.0)

    def test_linreg_r2_is_low_for_noise(self) -> None:
        rng = np.random.default_rng(11)
        _, r2 = rolling_linreg(series(list(rng.normal(0, 1, 200))), 20)
        assert r2.dropna().mean() < 0.5

    def test_hurst_exceeds_half_for_a_trend(self) -> None:
        values = list(4000 + np.arange(400, dtype="float64") * 0.5)
        assert hurst_variance_ratio(series(values), 100).iloc[-1] > 0.5


class TestNormalisation:
    def test_rolling_percentile_stays_in_unit_interval(self) -> None:
        rng = np.random.default_rng(5)
        result = rolling_percentile(series(list(rng.normal(0, 1, 500))), 100).dropna()
        assert result.between(0.0, 1.0).all()

    def test_rolling_percentile_only_sees_the_trailing_window(self) -> None:
        """A later spike must not change an earlier percentile."""
        base = list(np.arange(300, dtype="float64"))
        without = rolling_percentile(series(base), 50)
        with_spike = rolling_percentile(series([*base, 1e9]), 50)
        pd.testing.assert_series_equal(without, with_spike.iloc[:300])

    def test_zscore_of_a_flat_series_is_zero(self) -> None:
        assert rolling_zscore(series([7.0] * 40), 20).iloc[-1] == pytest.approx(0.0)

    def test_squash_maps_into_the_unit_interval(self) -> None:
        result = squash(series([0.0, 0.5, 1.0, 10.0, 1e6]), 1.0)
        assert result.between(0.0, 1.0).all()
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[-1] == pytest.approx(1.0)

    def test_squash_clips_negatives_to_zero(self) -> None:
        assert squash(series([-5.0]), 1.0).iloc[0] == pytest.approx(0.0)


class TestVolumePrimitives:
    def test_buy_pressure_is_bounded_and_signed(self) -> None:
        # Close at the high -> +1; close at the low -> -1.
        high, low = series([10.0, 10.0]), series([8.0, 8.0])
        close = series([10.0, 8.0])
        result = buy_pressure(high, low, close)
        assert result.iloc[0] == pytest.approx(1.0)
        assert result.iloc[1] == pytest.approx(-1.0)

    def test_buy_pressure_handles_a_zero_range_bar(self) -> None:
        flat = series([5.0])
        assert np.isfinite(buy_pressure(flat, flat, flat).iloc[0])
