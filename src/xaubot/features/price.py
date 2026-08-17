"""Candle geometry and multi-horizon returns.

Nothing here emits a raw price. Gold traded near 1,800 in 2023 and near 5,600
in this dataset's window; a model fed raw levels learns the level, not the
behaviour, and collapses the moment price leaves the training range. Every
feature is either a ratio (already scale-free) or normalised by ATR.

Multi-bar returns are additionally divided by ``sqrt(k)`` so that a 20-bar
return and a 1-bar return live on comparable scales -- otherwise the longer
horizons dominate purely through variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaubot.features._ta import rolling_zscore, sma
from xaubot.features.base import EPS, FeatureContext, FeatureSpec, Transform


class PriceTransform(Transform):
    """Bar shape, position, and normalised returns."""

    group = "price"

    def __init__(self, return_periods: tuple[int, ...], zscore_periods: tuple[int, ...]) -> None:
        self.return_periods = tuple(sorted(return_periods))
        self.zscore_periods = tuple(sorted(zscore_periods))

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        out: list[FeatureSpec] = [
            FeatureSpec("open_ret", self.group, 2, "log(open / previous close)"),
            FeatureSpec("high_ret", self.group, 2, "log(high / previous close)"),
            FeatureSpec("low_ret", self.group, 2, "log(low / previous close)"),
            FeatureSpec("close_ret", self.group, 2, "log(close / previous close)"),
            FeatureSpec("body_ret", self.group, 1, "(close - open) / close"),
            FeatureSpec("range_ret", self.group, 1, "(high - low) / close"),
            FeatureSpec("upper_wick", self.group, 1, "upper wick as a fraction of bar range"),
            FeatureSpec("lower_wick", self.group, 1, "lower wick as a fraction of bar range"),
            FeatureSpec("body_to_range", self.group, 1, "|body| as a fraction of bar range"),
            FeatureSpec("close_position", self.group, 1, "close position within the range, [0,1]"),
            FeatureSpec("high_low_range_atr", self.group, 15, "bar range in ATR units"),
            FeatureSpec("open_close_range_atr", self.group, 15, "|body| in ATR units"),
            FeatureSpec("gap_atr", self.group, 15, "(open - previous close) in ATR units"),
        ]
        out += [
            FeatureSpec(
                f"ret_{k}_atr",
                self.group,
                k + 15,
                f"{k}-bar return in ATR units, scaled by sqrt({k})",
            )
            for k in self.return_periods
        ]
        out += [
            FeatureSpec(f"zscore_close_{n}", self.group, n, f"close z-score over the trailing {n} bars")
            for n in self.zscore_periods
        ]
        out += [
            FeatureSpec("dist_sma_20_atr", self.group, 35, "close distance from SMA(20) in ATR units"),
        ]
        return tuple(out)

    def _compute(self, bars: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        open_ = bars["open"]
        high = bars["high"]
        low = bars["low"]
        close = bars["close"]
        atr = ctx.atr
        prev_close = close.shift(1)
        bar_range = high - low

        out = pd.DataFrame(index=bars.index)
        out["open_ret"] = np.log(open_ / prev_close)
        out["high_ret"] = np.log(high / prev_close)
        out["low_ret"] = np.log(low / prev_close)
        out["close_ret"] = np.log(close / prev_close)

        out["body_ret"] = (close - open_) / close
        out["range_ret"] = bar_range / close

        upper = high - np.maximum(open_, close)
        lower = np.minimum(open_, close) - low
        out["upper_wick"] = upper / (bar_range + EPS)
        out["lower_wick"] = lower / (bar_range + EPS)
        out["body_to_range"] = (close - open_).abs() / (bar_range + EPS)
        out["close_position"] = (close - low) / (bar_range + EPS)

        out["high_low_range_atr"] = bar_range / (atr + EPS)
        out["open_close_range_atr"] = (close - open_).abs() / (atr + EPS)
        out["gap_atr"] = (open_ - prev_close) / (atr + EPS)

        log_close = np.log(close)
        for k in self.return_periods:
            # sqrt(k) scaling puts every horizon on a comparable scale; without
            # it the longer horizons dominate through variance alone.
            out[f"ret_{k}_atr"] = (log_close - log_close.shift(k)) * close / (atr + EPS) / np.sqrt(k)

        for n in self.zscore_periods:
            out[f"zscore_close_{n}"] = rolling_zscore(close, n)

        out["dist_sma_20_atr"] = (close - sma(close, 20)) / (atr + EPS)

        return out
