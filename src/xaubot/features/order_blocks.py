"""Order blocks.

A bullish order block is the last down-close candle before a strong upward
displacement -- the idea being that whatever absorbed supply there may defend
the level again. Whether that idea is *true* is the model's problem; this
module's job is to detect the pattern consistently and without look-ahead.

The detection is inherently retrospective ("the last down candle *before* the
move"), which makes it a natural place to leak. The rule used here: the
displacement is measured at bar ``i``, and the order block is the qualifying
candle at ``i-1`` or earlier. The block therefore becomes visible only at bar
``i``, when the displacement that defines it has actually happened -- with an
``age`` that starts at the displacement bar, not the block candle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaubot.features._ta import squash
from xaubot.features.base import EPS, FeatureContext, FeatureSpec, Transform


class OrderBlockTransform(Transform):
    """Nearest live order block on each side, with mitigation tracking."""

    group = "order_blocks"

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        self.cfg = config

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        note = "detected at the displacement bar, so it is never visible before the move that defines it"
        out: list[FeatureSpec] = []
        for side in ("bull", "bear"):
            out += [
                FeatureSpec(
                    f"ob_{side}_exists", self.group, 3, f"a live {side}ish order block is tracked", note
                ),
                FeatureSpec(
                    f"ob_{side}_dist_atr", self.group, 3, "distance from close to the block edge", note
                ),
                FeatureSpec(f"ob_{side}_size_atr", self.group, 3, "block height in ATR units", note),
                FeatureSpec(
                    f"ob_{side}_age", self.group, 3, "bars since the displacement that created it", note
                ),
                FeatureSpec(
                    f"ob_{side}_strength",
                    self.group,
                    3,
                    "displacement size, freshness, and remaining unmitigated fraction, [0,1]",
                    note,
                ),
                FeatureSpec(
                    f"ob_{side}_mitigated", self.group, 3, "price has traded back into the block", note
                ),
                FeatureSpec(
                    f"ob_{side}_mitigation_pct", self.group, 3, "fraction of the block traded back", note
                ),
            ]
        out.append(
            FeatureSpec(
                "ob_confluence_fvg",
                self.group,
                3,
                "the displacement that created the block also left a gap, [0,1]",
                note,
            )
        )
        return tuple(out)

    def _compute(self, bars: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        cfg = self.cfg
        scan = _track_order_blocks(
            open_=bars["open"].to_numpy(),
            high=bars["high"].to_numpy(),
            low=bars["low"].to_numpy(),
            close=bars["close"].to_numpy(),
            atr=ctx.atr.to_numpy(),
            displacement_atr=cfg.displacement_atr,
            max_age=cfg.max_age_bars,
            max_tracked=cfg.max_tracked,
        )

        out = pd.DataFrame(index=bars.index)
        for side in ("bull", "bear"):
            out[f"ob_{side}_exists"] = scan[f"{side}_exists"]
            out[f"ob_{side}_dist_atr"] = scan[f"{side}_dist"]
            out[f"ob_{side}_size_atr"] = scan[f"{side}_size"]
            out[f"ob_{side}_age"] = scan[f"{side}_age"]
            out[f"ob_{side}_mitigated"] = scan[f"{side}_mitigated"]
            out[f"ob_{side}_mitigation_pct"] = scan[f"{side}_mitigation"]

            displacement = pd.Series(scan[f"{side}_displacement"], index=bars.index)
            age = pd.Series(scan[f"{side}_age"], index=bars.index)
            mitigation = pd.Series(scan[f"{side}_mitigation"], index=bars.index)
            freshness = 1.0 - (age / cfg.max_age_bars).clip(0.0, 1.0)
            out[f"ob_{side}_strength"] = (squash(displacement, 2.0) * freshness * (1.0 - mitigation)).fillna(
                0.0
            )

        out["ob_confluence_fvg"] = scan["confluence"]
        return out


def _track_order_blocks(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    displacement_atr: float,
    max_age: int,
    max_tracked: int,
) -> dict[str, np.ndarray]:
    """Forward pass detecting and ageing order blocks."""
    n = len(close)
    out: dict[str, np.ndarray] = {}
    for side in ("bull", "bear"):
        # Zero rather than NaN when no block exists. "No order block" is a real
        # state, not missing data, and the paired *_exists flag disambiguates
        # it. Encoding absence as NaN would push these columns past the dataset
        # stage's NaN threshold and silently drop the whole group.
        for field in ("exists", "mitigated", "mitigation", "dist", "size", "age", "displacement"):
            out[f"{side}_{field}"] = np.zeros(n)
    out["confluence"] = np.zeros(n)

    # block = [bottom, top, created_index, displacement_atr, extreme, had_gap]
    bull: list[list[float]] = []
    bear: list[list[float]] = []

    for i in range(n):
        current_atr = atr[i]
        if not np.isfinite(current_atr) or current_atr <= 0:
            continue

        if i >= 2:
            move = (close[i] - close[i - 1]) / current_atr

            if move >= displacement_atr:
                # Last bearish candle at or before i-1 becomes the bull block.
                for j in (i - 1, i - 2):
                    if j >= 0 and close[j] < open_[j]:
                        had_gap = 1.0 if (i >= 2 and low[i] > high[i - 2]) else 0.0
                        # Extreme starts at +inf, meaning "not yet penetrated".
                        # Seeding it with the block candle's own low would make
                        # the block instantly count as fully traded through.
                        bull.append([low[j], high[j], float(i), float(move), np.inf, had_gap])
                        break

            if move <= -displacement_atr:
                for j in (i - 1, i - 2):
                    if j >= 0 and close[j] > open_[j]:
                        had_gap = 1.0 if (i >= 2 and high[i] < low[i - 2]) else 0.0
                        bear.append([low[j], high[j], float(i), float(-move), -np.inf, had_gap])
                        break

        bull = [b for b in bull if (i - b[2]) <= max_age]
        bear = [b for b in bear if (i - b[2]) <= max_age]

        # Mitigation: how far price has traded back into the block so far.
        for block in bull:
            block[4] = min(block[4], low[i])
        for block in bear:
            block[4] = max(block[4], high[i])

        # Fully traded through: the block is spent, not merely touched.
        bull = [b for b in bull if b[4] > b[0]]
        bear = [b for b in bear if b[4] < b[1]]

        bull.sort(key=lambda b: abs(close[i] - b[1]))
        bear.sort(key=lambda b: abs(close[i] - b[0]))
        del bull[max_tracked:]
        del bear[max_tracked:]

        confluence = 0.0

        if bull:
            bottom, top, created, displacement, lowest, had_gap = bull[0]
            height = top - bottom
            out["bull_exists"][i] = 1.0
            out["bull_dist"][i] = (close[i] - top) / current_atr
            out["bull_size"][i] = height / current_atr
            out["bull_age"][i] = i - created
            out["bull_displacement"][i] = displacement
            mitigation = float(np.clip((top - lowest) / (height + EPS), 0.0, 1.0))
            out["bull_mitigation"][i] = mitigation
            out["bull_mitigated"][i] = 1.0 if mitigation > 0.0 else 0.0
            confluence = max(confluence, had_gap)

        if bear:
            bottom, top, created, displacement, highest, had_gap = bear[0]
            height = top - bottom
            out["bear_exists"][i] = 1.0
            out["bear_dist"][i] = (bottom - close[i]) / current_atr
            out["bear_size"][i] = height / current_atr
            out["bear_age"][i] = i - created
            out["bear_displacement"][i] = displacement
            mitigation = float(np.clip((highest - bottom) / (height + EPS), 0.0, 1.0))
            out["bear_mitigation"][i] = mitigation
            out["bear_mitigated"][i] = 1.0 if mitigation > 0.0 else 0.0
            confluence = max(confluence, had_gap)

        out["confluence"][i] = confluence

    return out
