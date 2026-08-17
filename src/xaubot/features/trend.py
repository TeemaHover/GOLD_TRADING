"""Trend direction, strength, and quality.

Indicators appear here as *features*, never as rules. An EMA cross is not a
signal in this system; ``ema_stack_score`` is a number the model may or may not
find predictive, and the walk-forward report will say which.

Two design choices worth noting:

- No EMA is fed to the model raw. What matters is the *distance* from price to
  the EMA (in ATR units) and the EMA's *slope* (also ATR-normalised), both of
  which are scale-free. The level itself is just a lagged price.
- ``efficiency_ratio`` and ``linreg_r2`` measure trend *quality* rather than
  direction. Gold spends a lot of time moving a long way without going
  anywhere, and separating those cases is more useful than another momentum
  oscillator.
"""

from __future__ import annotations

import pandas as pd

from xaubot.features._ta import (
    cci,
    directional_index,
    efficiency_ratio,
    ema,
    hurst_variance_ratio,
    macd_histogram,
    rolling_linreg,
    rsi,
    stochastic,
    williams_r,
)
from xaubot.features.base import EPS, FeatureContext, FeatureSpec, Transform


class TrendTransform(Transform):
    """EMA structure, ADX/DI, oscillators, and regression-based trend quality."""

    group = "trend"

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        self.cfg = config

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        cfg = self.cfg
        longest_ema = max(cfg.ema_periods)
        out: list[FeatureSpec] = []

        for period in cfg.ema_periods:
            out.append(
                FeatureSpec(
                    f"dist_ema_{period}_atr",
                    self.group,
                    period + 15,
                    f"(close - EMA{period}) in ATR units",
                )
            )
            out.append(
                FeatureSpec(
                    f"ema_slope_{period}",
                    self.group,
                    period + cfg.slope_lookback + 15,
                    f"EMA{period} slope over {cfg.slope_lookback} bars, ATR-normalised",
                )
            )

        fast, slow = sorted(cfg.ema_periods)[:2]
        out += [
            FeatureSpec(
                "ema_stack_score",
                self.group,
                longest_ema,
                "EMA ordering agreement in [-1,1]; +1 means fully stacked bullish",
            ),
            FeatureSpec(
                f"ema_{fast}_{slow}_spread_atr", self.group, slow + 15, "fast-slow EMA spread in ATR units"
            ),
            FeatureSpec(
                "ema_50_200_spread_atr", self.group, longest_ema + 15, "slow EMA spread in ATR units"
            ),
            FeatureSpec(f"adx_{cfg.adx_period}", self.group, cfg.adx_period * 3, "ADX trend strength"),
            FeatureSpec(
                f"di_plus_{cfg.adx_period}", self.group, cfg.adx_period * 2, "positive directional index"
            ),
            FeatureSpec(
                f"di_minus_{cfg.adx_period}", self.group, cfg.adx_period * 2, "negative directional index"
            ),
            FeatureSpec("di_diff", self.group, cfg.adx_period * 2, "normalised DI+ minus DI-, [-1,1]"),
            FeatureSpec("adx_slope_5", self.group, cfg.adx_period * 3 + 5, "5-bar change in ADX"),
            FeatureSpec("macd_hist_atr", self.group, 41, "MACD histogram in ATR units"),
            FeatureSpec(f"rsi_{cfg.rsi_period}", self.group, cfg.rsi_period * 2, "relative strength index"),
            FeatureSpec("rsi_slope_5", self.group, cfg.rsi_period * 2 + 5, "5-bar change in RSI"),
            FeatureSpec(
                "rsi_divergence_20",
                self.group,
                cfg.rsi_period * 2 + 20,
                "price makes a 20-bar extreme that RSI does not confirm, [-1,1]",
            ),
            FeatureSpec("stoch_k_14", self.group, 15, "stochastic %K"),
            FeatureSpec("stoch_d_3", self.group, 18, "stochastic %D"),
            FeatureSpec("cci_20_norm", self.group, 21, "CCI(20) scaled to roughly [-1,1]"),
            FeatureSpec("williams_r_14", self.group, 15, "Williams %R rescaled to [0,1]"),
        ]
        for period in cfg.linreg_periods:
            out.append(
                FeatureSpec(
                    f"linreg_slope_{period}_atr",
                    self.group,
                    period + 15,
                    f"least-squares slope over {period} bars, in ATR units per bar",
                )
            )
            out.append(
                FeatureSpec(
                    f"linreg_r2_{period}",
                    self.group,
                    period,
                    "R-squared of that fit: how cleanly the move trends, [0,1]",
                )
            )
        out += [
            FeatureSpec(
                f"efficiency_ratio_{cfg.efficiency_period}",
                self.group,
                cfg.efficiency_period + 1,
                "Kaufman efficiency ratio: net move / path length, [0,1]",
            ),
            FeatureSpec(
                f"hurst_vr_{cfg.hurst_period}",
                self.group,
                cfg.hurst_period + 10,
                "variance-ratio persistence estimate; >0.5 trending, <0.5 mean-reverting",
            ),
        ]
        return tuple(out)

    def _compute(self, bars: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        cfg = self.cfg
        high, low, close = bars["high"], bars["low"], bars["close"]
        atr = ctx.atr
        out = pd.DataFrame(index=bars.index)

        emas: dict[int, pd.Series] = {p: ema(close, p) for p in cfg.ema_periods}
        for period, series in emas.items():
            out[f"dist_ema_{period}_atr"] = (close - series) / (atr + EPS)
            out[f"ema_slope_{period}"] = (series - series.shift(cfg.slope_lookback)) / (
                cfg.slope_lookback * atr + EPS
            )

        out["ema_stack_score"] = self._stack_score(emas)

        fast, slow = sorted(cfg.ema_periods)[:2]
        out[f"ema_{fast}_{slow}_spread_atr"] = (emas[fast] - emas[slow]) / (atr + EPS)
        slowest_two = sorted(cfg.ema_periods)[-2:]
        out["ema_50_200_spread_atr"] = (emas[slowest_two[0]] - emas[slowest_two[1]]) / (atr + EPS)

        adx, di_plus, di_minus = directional_index(high, low, close, cfg.adx_period)
        out[f"adx_{cfg.adx_period}"] = adx
        out[f"di_plus_{cfg.adx_period}"] = di_plus
        out[f"di_minus_{cfg.adx_period}"] = di_minus
        out["di_diff"] = (di_plus - di_minus) / (di_plus + di_minus + EPS)
        out["adx_slope_5"] = adx - adx.shift(5)

        out["macd_hist_atr"] = macd_histogram(close) / (atr + EPS)

        rsi_series = rsi(close, cfg.rsi_period)
        out[f"rsi_{cfg.rsi_period}"] = rsi_series
        out["rsi_slope_5"] = rsi_series - rsi_series.shift(5)
        out["rsi_divergence_20"] = self._divergence(close, rsi_series, 20)

        k, d = stochastic(high, low, close, 14, 3)
        out["stoch_k_14"] = k
        out["stoch_d_3"] = d

        out["cci_20_norm"] = cci(high, low, close, 20) / 100.0
        out["williams_r_14"] = (williams_r(high, low, close, 14) + 100.0) / 100.0

        for period in cfg.linreg_periods:
            slope, r2 = rolling_linreg(close, period)
            out[f"linreg_slope_{period}_atr"] = slope / (atr + EPS)
            out[f"linreg_r2_{period}"] = r2

        out[f"efficiency_ratio_{cfg.efficiency_period}"] = efficiency_ratio(close, cfg.efficiency_period)
        out[f"hurst_vr_{cfg.hurst_period}"] = hurst_variance_ratio(close, cfg.hurst_period)

        return out

    @staticmethod
    def _stack_score(emas: dict[int, pd.Series]) -> pd.Series:
        """Agreement of the EMA ordering, in ``[-1, 1]``.

        +1 when every faster EMA sits above every slower one (textbook bullish
        stack), -1 when fully inverted, and near 0 when they are tangled -- which
        is the state that actually matters, because it is where trend-following
        setups fail.
        """
        periods = sorted(emas)
        pairs = [
            (emas[periods[i]] > emas[periods[j]]).astype("float64") * 2.0 - 1.0
            for i in range(len(periods))
            for j in range(i + 1, len(periods))
        ]
        stacked = pd.concat(pairs, axis=1)
        score = stacked.mean(axis=1)
        # Preserve NaN during warmup instead of scoring on partial information.
        any_nan = pd.concat([emas[p].isna() for p in periods], axis=1).any(axis=1)
        return score.mask(any_nan)

    @staticmethod
    def _divergence(price: pd.Series, oscillator: pd.Series, period: int) -> pd.Series:
        """Regular divergence between price extremes and oscillator extremes.

        +1 bullish (price at a new low, oscillator not), -1 bearish. Uses only
        trailing rolling extremes, so a divergence is reported at the bar where
        it becomes visible -- not backdated to the pivot, which is the usual way
        this indicator smuggles in future information.
        """
        price_low = price <= price.rolling(period, min_periods=period).min()
        price_high = price >= price.rolling(period, min_periods=period).max()
        osc_low = oscillator <= oscillator.rolling(period, min_periods=period).min()
        osc_high = oscillator >= oscillator.rolling(period, min_periods=period).max()

        bullish = (price_low & ~osc_low).astype("float64")
        bearish = (price_high & ~osc_high).astype("float64")
        return (bullish - bearish).mask(price.rolling(period, min_periods=period).min().isna())
