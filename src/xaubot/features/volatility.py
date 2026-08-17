"""Volatility level, shape, and regime.

Volatility is the single most useful conditioning variable in this system: it
sets stop distances, it gates whether a setup is tradable at all, and it
determines whether a given expected move clears the cost hurdle. It is also the
most regime-dependent thing about gold, which is why almost everything here is
expressed as a *percentile against trailing history* rather than as a level --
an ATR of $8 means something completely different in a quiet January than in a
news-driven March.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaubot.features._ta import (
    atr as atr_fn,
)
from xaubot.features._ta import (
    bollinger_width,
    garman_klass_vol,
    keltner_width,
    parkinson_vol,
    realized_vol,
    rolling_percentile,
    sma,
)
from xaubot.features.base import EPS, FeatureContext, FeatureSpec, Transform


class VolatilityTransform(Transform):
    """ATR family, realised-volatility estimators, expansion, and regime."""

    group = "volatility"

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        self.cfg = config

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        cfg = self.cfg
        longest_pct = max(cfg.percentile_windows)
        out: list[FeatureSpec] = []

        for period in cfg.atr_periods:
            out.append(FeatureSpec(f"atr_{period}", self.group, period + 1, f"ATR({period}), Wilder"))
        out.append(FeatureSpec("atr_pct", self.group, 15, "ATR as a fraction of price"))
        out.append(
            FeatureSpec(
                "atr_ratio_fast_slow",
                self.group,
                max(cfg.atr_periods) + 1,
                "fast ATR / slow ATR: >1 means volatility is expanding",
            )
        )
        for window in cfg.percentile_windows:
            out.append(
                FeatureSpec(
                    f"atr_percentile_{window}",
                    self.group,
                    window,
                    f"rank of ATR within the trailing {window} bars, [0,1]",
                    pit_note="trailing-window rank; a global rank would leak the future",
                )
            )
        for period in cfg.realized_vol_periods:
            out.append(
                FeatureSpec(f"realized_vol_{period}", self.group, period + 1, f"{period}-bar realised vol")
            )
        out += [
            FeatureSpec("parkinson_vol_20", self.group, 21, "Parkinson high-low volatility estimator"),
            FeatureSpec("garman_klass_vol_20", self.group, 21, "Garman-Klass OHLC volatility estimator"),
            FeatureSpec("vol_of_vol_50", self.group, 71, "volatility of realised volatility"),
            FeatureSpec("range_expansion", self.group, 21, "bar range / mean range over 20 bars"),
            FeatureSpec("range_compression", self.group, 51, "mean range(5) / mean range(50)"),
            FeatureSpec("nr7", self.group, 7, "1 if this is the narrowest range of the last 7 bars"),
            FeatureSpec("wr7", self.group, 7, "1 if this is the widest range of the last 7 bars"),
            FeatureSpec(f"bb_width_{cfg.bb_period}", self.group, cfg.bb_period, "Bollinger width / mid"),
            FeatureSpec(
                f"bb_width_percentile_{cfg.bb_percentile_window}",
                self.group,
                cfg.bb_percentile_window,
                "trailing rank of Bollinger width, [0,1]",
            ),
            FeatureSpec(
                "squeeze_flag",
                self.group,
                cfg.bb_period + 1,
                "1 when Bollinger width is inside Keltner width (compression)",
            ),
            FeatureSpec(
                "vol_regime",
                self.group,
                longest_pct,
                "ordinal volatility bucket: 0 LOW, 1 NORMAL, 2 HIGH, 3 EXTREME",
            ),
        ]
        out += [
            FeatureSpec(
                f"vol_regime_is_{name}",
                self.group,
                longest_pct,
                f"one-hot: volatility regime is {name.upper()}",
            )
            for name in ("low", "normal", "high", "extreme")
        ]
        return tuple(out)

    def _compute(self, bars: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        cfg = self.cfg
        open_, high, low, close = bars["open"], bars["high"], bars["low"], bars["close"]
        out = pd.DataFrame(index=bars.index)

        atrs: dict[int, pd.Series] = {}
        for period in cfg.atr_periods:
            atrs[period] = atr_fn(high, low, close, period)
            out[f"atr_{period}"] = atrs[period]

        primary = atrs[cfg.primary_atr_period]
        out["atr_pct"] = primary / (close + EPS)

        fast, slow = min(cfg.atr_periods), max(cfg.atr_periods)
        out["atr_ratio_fast_slow"] = atrs[fast] / (atrs[slow] + EPS)

        for window in cfg.percentile_windows:
            out[f"atr_percentile_{window}"] = rolling_percentile(primary, window)

        bars_per_day = ctx.timeframe.bars_per_day
        for period in cfg.realized_vol_periods:
            out[f"realized_vol_{period}"] = realized_vol(close, period, bars_per_day)

        out["parkinson_vol_20"] = parkinson_vol(high, low, 20)
        out["garman_klass_vol_20"] = garman_klass_vol(open_, high, low, close, 20)

        rv_short = realized_vol(close, 20, bars_per_day)
        out["vol_of_vol_50"] = rv_short.rolling(50, min_periods=50).std(ddof=0)

        bar_range = high - low
        out["range_expansion"] = bar_range / (sma(bar_range, 20) + EPS)
        out["range_compression"] = sma(bar_range, 5) / (sma(bar_range, 50) + EPS)

        # min_periods=7 so the first six bars are NaN rather than being judged
        # against a partial window.
        out["nr7"] = (bar_range <= bar_range.rolling(7, min_periods=7).min()).astype("float64")
        out["wr7"] = (bar_range >= bar_range.rolling(7, min_periods=7).max()).astype("float64")

        bb = bollinger_width(close, cfg.bb_period)
        out[f"bb_width_{cfg.bb_period}"] = bb
        out[f"bb_width_percentile_{cfg.bb_percentile_window}"] = rolling_percentile(
            bb, cfg.bb_percentile_window
        )
        out["squeeze_flag"] = (bb < keltner_width(high, low, close, cfg.bb_period)).astype("float64")

        regime = self._regime(out[f"atr_percentile_{max(cfg.percentile_windows)}"])
        out["vol_regime"] = regime
        for value, name in enumerate(("low", "normal", "high", "extreme")):
            out[f"vol_regime_is_{name}"] = (regime == value).astype("float64")

        return out

    def _regime(self, percentile: pd.Series) -> pd.Series:
        """Bucket the ATR percentile into an ordinal regime.

        Kept ordinal *and* one-hot: the ordinal encoding carries the natural
        ordering for tree models, the one-hot lets a network learn a non-monotone
        response (both dead and chaotic markets are untradable, for different
        reasons).
        """
        cfg = self.cfg
        regime = pd.Series(np.full(len(percentile), np.nan), index=percentile.index)
        valid = percentile.notna()
        regime[valid & (percentile < cfg.regime_low_pct)] = 0.0
        regime[valid & (percentile >= cfg.regime_low_pct) & (percentile < cfg.regime_high_pct)] = 1.0
        regime[valid & (percentile >= cfg.regime_high_pct) & (percentile < cfg.regime_extreme_pct)] = 2.0
        regime[valid & (percentile >= cfg.regime_extreme_pct)] = 3.0
        return regime
