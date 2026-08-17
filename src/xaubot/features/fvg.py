"""Fair value gaps (three-bar imbalances).

A bullish FVG exists when bar ``i``'s low sits above bar ``i-2``'s high: price
moved so fast that a band of prices was never traded through. The pattern is
only complete at bar ``i``, so that is where it is stamped -- never backdated
to ``i-1`` or ``i-2``, which would make the gap visible before the bar that
created it.

Gaps are tracked as living objects: they age, they partially fill, and they
expire. The features describe the nearest unfilled gap on each side, because
that is the one price is actually likely to react to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaubot.features._ta import squash
from xaubot.features.base import EPS, FeatureContext, FeatureSpec, Transform


class FvgTransform(Transform):
    """Nearest unfilled fair value gap on each side, plus population counts."""

    group = "fvg"

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        self.cfg = config

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        note = "a 3-bar gap is stamped at the third bar, where the pattern completes"
        out: list[FeatureSpec] = []
        for side in ("bull", "bear"):
            out += [
                FeatureSpec(
                    f"fvg_{side}_exists", self.group, 3, f"an unfilled {side}ish FVG is tracked", note
                ),
                FeatureSpec(f"fvg_{side}_size_atr", self.group, 3, "gap height in ATR units", note),
                FeatureSpec(
                    f"fvg_{side}_dist_atr", self.group, 3, "distance from close to the gap edge", note
                ),
                FeatureSpec(f"fvg_{side}_age", self.group, 3, "bars since the gap formed", note),
                FeatureSpec(
                    f"fvg_{side}_fill_pct", self.group, 3, "fraction of the gap traded back through", note
                ),
                FeatureSpec(
                    f"fvg_{side}_strength",
                    self.group,
                    3,
                    "size, freshness, and remaining fill combined into [0,1]",
                    note,
                ),
            ]
        out += [
            FeatureSpec("fvg_count_bull", self.group, 3, "unfilled bullish gaps currently tracked", note),
            FeatureSpec("fvg_count_bear", self.group, 3, "unfilled bearish gaps currently tracked", note),
            FeatureSpec(
                "fvg_imbalance", self.group, 3, "(bull - bear) / (bull + bear) gap count, [-1,1]", note
            ),
        ]
        return tuple(out)

    def _compute(self, bars: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        cfg = self.cfg
        high = bars["high"].to_numpy()
        low = bars["low"].to_numpy()
        close = bars["close"].to_numpy()
        atr = ctx.atr.to_numpy()

        scan = _track_gaps(
            high=high,
            low=low,
            close=close,
            atr=atr,
            min_size_atr=cfg.min_size_atr,
            max_age=cfg.max_age_bars,
            max_tracked=cfg.max_tracked,
        )

        out = pd.DataFrame(index=bars.index)
        for side in ("bull", "bear"):
            out[f"fvg_{side}_exists"] = scan[f"{side}_exists"]
            out[f"fvg_{side}_size_atr"] = scan[f"{side}_size"]
            out[f"fvg_{side}_dist_atr"] = scan[f"{side}_dist"]
            out[f"fvg_{side}_age"] = scan[f"{side}_age"]
            out[f"fvg_{side}_fill_pct"] = scan[f"{side}_fill"]

            size = pd.Series(scan[f"{side}_size"], index=bars.index)
            age = pd.Series(scan[f"{side}_age"], index=bars.index)
            fill = pd.Series(scan[f"{side}_fill"], index=bars.index)
            freshness = 1.0 - (age / cfg.max_age_bars).clip(0.0, 1.0)
            out[f"fvg_{side}_strength"] = (squash(size, 1.0) * freshness * (1.0 - fill)).fillna(0.0)

        out["fvg_count_bull"] = scan["bull_count"]
        out["fvg_count_bear"] = scan["bear_count"]
        total = scan["bull_count"] + scan["bear_count"]
        out["fvg_imbalance"] = np.divide(
            scan["bull_count"] - scan["bear_count"],
            total,
            out=np.zeros_like(total),
            where=total > 0,
        )

        return out


def _track_gaps(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    min_size_atr: float,
    max_age: int,
    max_tracked: int,
) -> dict[str, np.ndarray]:
    """Forward pass maintaining live gap objects.

    Each gap is ``[bottom, top, created_index, extreme]`` where ``extreme``
    tracks how far price has traded back into it so far. State only ever moves
    forward, so a gap's fill percentage at bar ``i`` reflects bars ``<= i``.
    """
    n = len(close)
    # Zero rather than NaN when no gap exists. "No gap" is a real state, not
    # missing data, and the paired *_exists flag disambiguates it. Encoding
    # absence as NaN would push these columns past the dataset stage's NaN
    # threshold and drop genuinely informative features.
    fields = ("exists", "size", "dist", "age", "fill", "count")
    out = {f"{side}_{field}": np.zeros(n) for side in ("bull", "bear") for field in fields}

    bull: list[list[float]] = []  # [bottom, top, created_index, lowest_since]
    bear: list[list[float]] = []  # [bottom, top, created_index, highest_since]

    for i in range(n):
        current_atr = atr[i]
        if not np.isfinite(current_atr) or current_atr <= 0:
            continue

        # --- detect a new gap completed by this bar ------------------------
        if i >= 2:
            if low[i] > high[i - 2]:
                size = low[i] - high[i - 2]
                if size >= min_size_atr * current_atr:
                    bull.append([high[i - 2], low[i], float(i), low[i]])
            if high[i] < low[i - 2]:
                size = low[i - 2] - high[i]
                if size >= min_size_atr * current_atr:
                    bear.append([high[i], low[i - 2], float(i), high[i]])

        # --- age out and fill in --------------------------------------------
        bull = [gap for gap in bull if (i - gap[2]) <= max_age]
        bear = [gap for gap in bear if (i - gap[2]) <= max_age]

        for gap in bull:
            gap[3] = min(gap[3], low[i])
        for gap in bear:
            gap[3] = max(gap[3], high[i])

        # A bullish gap is fully filled once price trades back below its floor.
        bull = [gap for gap in bull if gap[3] > gap[0]]
        bear = [gap for gap in bear if gap[3] < gap[1]]

        # Keep only the gaps nearest to price; distant ones are noise.
        bull.sort(key=lambda gap: abs(close[i] - gap[1]))
        bear.sort(key=lambda gap: abs(close[i] - gap[0]))
        del bull[max_tracked:]
        del bear[max_tracked:]

        out["bull_count"][i] = len(bull)
        out["bear_count"][i] = len(bear)

        if bull:
            bottom, top, created, lowest = bull[0]
            height = top - bottom
            out["bull_exists"][i] = 1.0
            out["bull_size"][i] = height / current_atr
            out["bull_dist"][i] = (close[i] - top) / current_atr
            out["bull_age"][i] = i - created
            out["bull_fill"][i] = float(np.clip((top - lowest) / (height + EPS), 0.0, 1.0))

        if bear:
            bottom, top, created, highest = bear[0]
            height = top - bottom
            out["bear_exists"][i] = 1.0
            out["bear_size"][i] = height / current_atr
            out["bear_dist"][i] = (bottom - close[i]) / current_atr
            out["bear_age"][i] = i - created
            out["bear_fill"][i] = float(np.clip((highest - bottom) / (height + EPS), 0.0, 1.0))

    return out
