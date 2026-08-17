"""Market structure: swings, BOS, MSS, and the HH/HL/LH/LL sequence.

**The confirmation-delay rule.** A fractal swing high at bar ``i`` requires
``k`` bars on each side to be higher-free. It is therefore not knowable until
bar ``i + k``. Almost every published implementation marks the swing at bar
``i``, which backdates knowledge by ``k`` bars and makes every structure
feature clairvoyant -- and because the leak is small and local, it survives
casual inspection while inflating backtest results substantially.

Here, a pivot at bar ``i`` first becomes visible at bar ``i + k``. The
implementation makes this structural rather than careful:

    confirmed[j] = high.shift(k)[j] >= high.rolling(2k+1).max()[j]

evaluated at bar ``j``, which reads bars ``j-2k .. j`` only. The pivot it
confirms sits at ``j - k``. ``swing_*_age`` counts from the pivot, so the age
of a freshly confirmed swing is ``k``, never 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaubot.features._ta import squash
from xaubot.features.base import EPS, FeatureContext, FeatureSpec, Transform


class StructureTransform(Transform):
    """Confirmed swings and the structural events derived from them."""

    group = "structure"

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        self.cfg = config

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        cfg = self.cfg
        k = cfg.swing_k
        confirm = 2 * k + 1
        note = f"pivot at bar i is first visible at bar i+{k}; ages count from the pivot"
        long_lookback = max(cfg.range_period, confirm * cfg.structure_memory)

        return (
            FeatureSpec(
                "swing_high_flag", self.group, confirm, "a swing high was confirmed on this bar", note
            ),
            FeatureSpec("swing_low_flag", self.group, confirm, "a swing low was confirmed on this bar", note),
            FeatureSpec(
                "last_swing_high_dist_atr", self.group, confirm, "distance to last confirmed swing high", note
            ),
            FeatureSpec(
                "last_swing_low_dist_atr", self.group, confirm, "distance to last confirmed swing low", note
            ),
            FeatureSpec("last_swing_high_age", self.group, confirm, "bars since that swing high pivot", note),
            FeatureSpec("last_swing_low_age", self.group, confirm, "bars since that swing low pivot", note),
            FeatureSpec(
                "hh", self.group, confirm * 2, "latest confirmed swing high exceeded the previous", note
            ),
            FeatureSpec(
                "hl", self.group, confirm * 2, "latest confirmed swing low exceeded the previous", note
            ),
            FeatureSpec(
                "lh", self.group, confirm * 2, "latest confirmed swing high undercut the previous", note
            ),
            FeatureSpec(
                "ll", self.group, confirm * 2, "latest confirmed swing low undercut the previous", note
            ),
            FeatureSpec(
                "structure_direction", self.group, confirm * 2, "-1 bearish, 0 unclear, +1 bullish", note
            ),
            FeatureSpec(
                "structure_strength",
                self.group,
                long_lookback,
                "consistency of recent structure events, [0,1]",
                note,
            ),
            FeatureSpec(
                "bos_bull", self.group, confirm, "close broke above the last confirmed swing high", note
            ),
            FeatureSpec(
                "bos_bear", self.group, confirm, "close broke below the last confirmed swing low", note
            ),
            FeatureSpec(
                "bos_strength", self.group, confirm, "break distance in ATR units, squashed to [0,1]", note
            ),
            FeatureSpec(
                "bos_age", self.group, confirm, "bars since the most recent break of structure", note
            ),
            FeatureSpec(
                "mss_bull", self.group, confirm * 2, "bullish break against the prevailing structure", note
            ),
            FeatureSpec(
                "mss_bear", self.group, confirm * 2, "bearish break against the prevailing structure", note
            ),
            FeatureSpec("mss_strength", self.group, confirm * 2, "strength of that shift, [0,1]", note),
            FeatureSpec(
                "mss_age", self.group, confirm * 2, "bars since the most recent structure shift", note
            ),
            FeatureSpec(
                "range_high_dist_atr", self.group, cfg.range_period, "distance to the rolling range high"
            ),
            FeatureSpec(
                "range_low_dist_atr", self.group, cfg.range_period, "distance to the rolling range low"
            ),
            FeatureSpec("range_width_atr", self.group, cfg.range_period, "rolling range width in ATR units"),
            FeatureSpec(
                "range_position", self.group, cfg.range_period, "position within the rolling range, [0,1]"
            ),
            FeatureSpec(
                "consolidation_score",
                self.group,
                cfg.range_period,
                "how range-bound the market is, [0,1]: narrow range plus centred price",
            ),
        )

    def _compute(self, bars: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        cfg = self.cfg
        k = cfg.swing_k
        window = 2 * k + 1

        high, low, close = bars["high"], bars["low"], bars["close"]
        atr = ctx.atr

        # A pivot at bar j-k is confirmed at bar j, reading bars j-2k..j only.
        rolling_high = high.rolling(window, min_periods=window).max()
        rolling_low = low.rolling(window, min_periods=window).min()
        confirmed_high = (high.shift(k) >= rolling_high) & rolling_high.notna()
        confirmed_low = (low.shift(k) <= rolling_low) & rolling_low.notna()

        scan = _scan_structure(
            confirmed_high=confirmed_high.to_numpy(),
            confirmed_low=confirmed_low.to_numpy(),
            pivot_high=high.shift(k).to_numpy(),
            pivot_low=low.shift(k).to_numpy(),
            close=close.to_numpy(),
            k=k,
            memory=cfg.structure_memory,
        )

        out = pd.DataFrame(index=bars.index)
        out["swing_high_flag"] = confirmed_high.astype("float64")
        out["swing_low_flag"] = confirmed_low.astype("float64")

        last_high = pd.Series(scan["last_high_price"], index=bars.index)
        last_low = pd.Series(scan["last_low_price"], index=bars.index)
        out["last_swing_high_dist_atr"] = (close - last_high) / (atr + EPS)
        out["last_swing_low_dist_atr"] = (close - last_low) / (atr + EPS)
        out["last_swing_high_age"] = scan["last_high_age"]
        out["last_swing_low_age"] = scan["last_low_age"]

        for name in ("hh", "hl", "lh", "ll"):
            out[name] = scan[name]

        out["structure_direction"] = scan["direction"]
        out["structure_strength"] = scan["strength"]

        out["bos_bull"] = scan["bos_bull"]
        out["bos_bear"] = scan["bos_bear"]
        out["bos_strength"] = squash(pd.Series(scan["bos_distance"], index=bars.index) / (atr + EPS), 1.0)
        out["bos_age"] = scan["bos_age"]

        out["mss_bull"] = scan["mss_bull"]
        out["mss_bear"] = scan["mss_bear"]
        out["mss_strength"] = squash(pd.Series(scan["mss_distance"], index=bars.index) / (atr + EPS), 1.0)
        out["mss_age"] = scan["mss_age"]

        period = cfg.range_period
        range_high = high.rolling(period, min_periods=period).max()
        range_low = low.rolling(period, min_periods=period).min()
        width = range_high - range_low
        out["range_high_dist_atr"] = (range_high - close) / (atr + EPS)
        out["range_low_dist_atr"] = (close - range_low) / (atr + EPS)
        out["range_width_atr"] = width / (atr + EPS)
        position = (close - range_low) / (width + EPS)
        out["range_position"] = position

        # Range-bound means both a narrow range *and* price sitting inside it;
        # a narrow range with price pinned to one edge is a coiling breakout.
        narrow = 1.0 - squash(width / (atr * 10.0 + EPS), 1.0)
        centred = 1.0 - (position - 0.5).abs() * 2.0
        out["consolidation_score"] = (narrow * centred).clip(0.0, 1.0)

        return out


def _scan_structure(
    confirmed_high: np.ndarray,
    confirmed_low: np.ndarray,
    pivot_high: np.ndarray,
    pivot_low: np.ndarray,
    close: np.ndarray,
    k: int,
    memory: int,
) -> dict[str, np.ndarray]:
    """Single forward pass over confirmed pivots.

    A loop rather than vectorised operations because the state is genuinely
    sequential: whether a break counts as a *shift* depends on the structural
    direction implied by everything before it. Written as one pass so that no
    step can accidentally read ahead -- index ``i`` only ever touches state
    accumulated from indices ``< i`` plus bar ``i`` itself.
    """
    n = len(close)
    nan = np.nan

    out = {
        name: np.full(n, nan)
        for name in (
            "last_high_price",
            "last_low_price",
            "last_high_age",
            "last_low_age",
            "hh",
            "hl",
            "lh",
            "ll",
            "direction",
            "strength",
            "bos_bull",
            "bos_bear",
            "bos_distance",
            "bos_age",
            "mss_bull",
            "mss_bear",
            "mss_distance",
            "mss_age",
        )
    }

    last_high = nan
    prev_high = nan
    last_low = nan
    prev_low = nan
    last_high_index = -1
    last_low_index = -1

    is_hh = is_hl = is_lh = is_ll = 0.0
    direction = 0.0
    events: list[float] = []  # +1 bullish structural event, -1 bearish

    bos_index = -1
    bos_dir = 0.0
    bos_distance = nan
    mss_index = -1
    mss_dir = 0.0
    mss_distance = nan

    for i in range(n):
        # --- register newly confirmed pivots (pivot sits at i-k) ------------
        if confirmed_high[i] and not np.isnan(pivot_high[i]):
            prev_high = last_high
            last_high = pivot_high[i]
            last_high_index = i - k
            if not np.isnan(prev_high):
                is_hh = 1.0 if last_high > prev_high else 0.0
                is_lh = 1.0 if last_high < prev_high else 0.0
                events.append(1.0 if last_high > prev_high else -1.0)

        if confirmed_low[i] and not np.isnan(pivot_low[i]):
            prev_low = last_low
            last_low = pivot_low[i]
            last_low_index = i - k
            if not np.isnan(prev_low):
                is_hl = 1.0 if last_low > prev_low else 0.0
                is_ll = 1.0 if last_low < prev_low else 0.0
                events.append(1.0 if last_low > prev_low else -1.0)

        if len(events) > memory:
            del events[:-memory]

        # --- structural direction and its consistency -----------------------
        if is_hh and is_hl:
            direction = 1.0
        elif is_lh and is_ll:
            direction = -1.0
        elif len(events) >= 2:
            direction = float(np.sign(sum(events)))
        else:
            direction = 0.0

        # Strength is the net agreement of recent events, so alternating
        # HH/LL chop scores near zero even though events keep arriving.
        strength = abs(sum(events)) / len(events) if events else nan

        # --- break of structure --------------------------------------------
        broke_up = (not np.isnan(last_high)) and close[i] > last_high
        broke_down = (not np.isnan(last_low)) and close[i] < last_low

        if broke_up:
            bos_distance = close[i] - last_high
            if bos_dir <= 0:  # a break opposing the prior break direction
                mss_index, mss_dir, mss_distance = i, 1.0, bos_distance
            bos_index, bos_dir = i, 1.0
        elif broke_down:
            bos_distance = last_low - close[i]
            if bos_dir >= 0:
                mss_index, mss_dir, mss_distance = i, -1.0, bos_distance
            bos_index, bos_dir = i, -1.0

        # --- emit ------------------------------------------------------------
        out["last_high_price"][i] = last_high
        out["last_low_price"][i] = last_low
        out["last_high_age"][i] = (i - last_high_index) if last_high_index >= 0 else nan
        out["last_low_age"][i] = (i - last_low_index) if last_low_index >= 0 else nan
        out["hh"][i] = is_hh
        out["hl"][i] = is_hl
        out["lh"][i] = is_lh
        out["ll"][i] = is_ll
        out["direction"][i] = direction
        out["strength"][i] = strength

        out["bos_bull"][i] = 1.0 if broke_up else 0.0
        out["bos_bear"][i] = 1.0 if broke_down else 0.0
        out["bos_distance"][i] = bos_distance if bos_index >= 0 else nan
        out["bos_age"][i] = (i - bos_index) if bos_index >= 0 else nan

        out["mss_bull"][i] = 1.0 if (mss_index == i and mss_dir > 0) else 0.0
        out["mss_bear"][i] = 1.0 if (mss_index == i and mss_dir < 0) else 0.0
        out["mss_distance"][i] = mss_distance if mss_index >= 0 else nan
        out["mss_age"][i] = (i - mss_index) if mss_index >= 0 else nan

    return out
