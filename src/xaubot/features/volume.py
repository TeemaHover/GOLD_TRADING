"""Volume and activity features.

The important one here is ``relative_volume_tod``. A plain volume ratio on 5m
gold is dominated by time of day -- 03:00 UTC is always quiet and 13:30 UTC is
always busy -- so a model handed a raw ratio mostly learns a clock. Comparing
each bar against the *same minute of day* over the trailing weeks isolates the
part that is actually informative: unusual activity for this time of day.

All of these depend on what the feed's volume column means. For MT5 XAUUSD it
is a tick count, not traded contracts (see docs/DATA_CONTRACT.md), which makes
it a usable activity proxy but not comparable across brokers. The whole group
can be switched off with ``features.volume.enabled=false`` for cross-feed
robustness testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaubot.features._ta import buy_pressure, on_balance_volume, rolling_zscore, sma
from xaubot.features.base import EPS, FeatureContext, FeatureSpec, Transform


class VolumeTransform(Transform):
    """Relative volume, expansion/compression, and order-flow proxies."""

    group = "volume"

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        self.cfg = config

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        cfg = self.cfg
        longest = max(cfg.ma_periods)
        out: list[FeatureSpec] = []
        for period in cfg.ma_periods:
            out.append(FeatureSpec(f"volume_ma_{period}", self.group, period, f"{period}-bar mean volume"))
        out += [
            FeatureSpec("volume_ratio", self.group, min(cfg.ma_periods), "volume / its short moving average"),
            FeatureSpec(
                "relative_volume_tod",
                self.group,
                288 * cfg.time_of_day_days,
                f"volume vs the median at this minute-of-day over the trailing {cfg.time_of_day_days} days",
                pit_note="per-group median is shifted by one occurrence, excluding the current bar",
            ),
            FeatureSpec(
                f"volume_zscore_{cfg.zscore_period}",
                self.group,
                cfg.zscore_period,
                "volume z-score against its trailing window",
            ),
            FeatureSpec("volume_expansion", self.group, longest, "mean volume(5) / mean volume(50)"),
            FeatureSpec("volume_compression", self.group, longest, "inverse of volume_expansion"),
            FeatureSpec("buy_pressure", self.group, 1, "close position within the bar range, [-1,1]"),
            FeatureSpec(
                "volume_weighted_buy_pressure",
                self.group,
                min(cfg.ma_periods),
                "buy_pressure scaled by relative volume",
            ),
            FeatureSpec(
                "cvd_proxy_20", self.group, 20 + cfg.zscore_period, "z-scored cumulative delta proxy"
            ),
            FeatureSpec(
                "cvd_proxy_50", self.group, 50 + cfg.zscore_period, "z-scored cumulative delta proxy"
            ),
            FeatureSpec("obv_slope_20", self.group, 21, "normalised on-balance-volume slope"),
            FeatureSpec("dist_vwap_atr", self.group, 1, "distance from session VWAP in ATR units"),
            FeatureSpec(
                "volume_is_missing", self.group, 1, "1 when the feed supplied no volume for this bar"
            ),
        ]
        return tuple(out)

    def _compute(self, bars: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        cfg = self.cfg
        high, low, close = bars["high"], bars["low"], bars["close"]
        volume = bars["volume"]
        out = pd.DataFrame(index=bars.index)

        out["volume_is_missing"] = volume.isna().astype("float64")
        volume = volume.fillna(0.0)

        for period in cfg.ma_periods:
            out[f"volume_ma_{period}"] = sma(volume, period)

        short_ma = out[f"volume_ma_{min(cfg.ma_periods)}"]
        out["volume_ratio"] = volume / (short_ma + EPS)
        out["relative_volume_tod"] = self._relative_volume_by_time_of_day(volume, cfg.time_of_day_days)
        out[f"volume_zscore_{cfg.zscore_period}"] = rolling_zscore(volume, cfg.zscore_period)

        expansion = sma(volume, 5) / (sma(volume, max(cfg.ma_periods)) + EPS)
        out["volume_expansion"] = expansion
        out["volume_compression"] = 1.0 / (expansion + EPS)

        pressure = buy_pressure(high, low, close)
        out["buy_pressure"] = pressure
        out["volume_weighted_buy_pressure"] = pressure * out["volume_ratio"]

        delta = pressure * volume
        for period in (20, 50):
            out[f"cvd_proxy_{period}"] = rolling_zscore(
                delta.rolling(period, min_periods=period).sum(), cfg.zscore_period
            )

        obv = on_balance_volume(close, volume)
        out["obv_slope_20"] = (obv - obv.shift(20)) / (volume.rolling(20, min_periods=20).sum() + EPS)

        out["dist_vwap_atr"] = (close - self._session_vwap(bars, ctx)) / (ctx.atr + EPS)

        return out

    @staticmethod
    def _relative_volume_by_time_of_day(volume: pd.Series, days: int) -> pd.Series:
        """Volume relative to the typical volume at this minute of day.

        The ``shift(1)`` inside each time-of-day group is what makes this
        point-in-time correct: without it the median would include the current
        bar, so an unusually large bar would partly normalise itself away.
        """
        minute_of_day = volume.index.hour * 60 + volume.index.minute
        grouped = volume.groupby(minute_of_day, group_keys=False)
        baseline = grouped.apply(
            lambda group: group.shift(1).rolling(days, min_periods=max(3, days // 4)).median()
        )
        baseline = baseline.reindex(volume.index)
        return volume / (baseline + EPS)

    @staticmethod
    def _session_vwap(bars: pd.DataFrame, ctx: FeatureContext) -> pd.Series:
        """Volume-weighted average price, reset at each session boundary.

        Uses an expanding sum *within* the current session only, so it never
        borrows from a session that has not happened yet.
        """
        typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
        volume = bars["volume"].fillna(0.0)
        session_id = ctx.sessions["session_id"].reindex(bars.index)

        weighted = (typical * volume).groupby(session_id).cumsum()
        total = volume.groupby(session_id).cumsum()
        # Fall back to typical price when a session has reported no volume yet.
        return np.where(total > 0, weighted / (total + EPS), typical)
