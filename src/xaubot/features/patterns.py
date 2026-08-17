"""Candle patterns as continuous strengths rather than binary flags.

Binary pattern flags throw away most of the information. A bullish engulfing
bar that swallows the previous body by 1.05x and one that swallows it by 4x are
not the same event, but ``+1`` says they are. Every pattern here emits a
strength in ``[0, 1]``, built from the components that make the pattern more or
less convincing.

Directional patterns emit a separate ``_bull_`` and ``_bear_`` column rather
than one signed column. Both scale the same way, and the model never has to
learn that "negative strength" means a different pattern rather than a weaker
one.

Scaling uses ``tanh`` against an ATR-relative quantity, never a min-max over
the series. A global min-max is leakage; a trailing min-max makes an identical
candle score differently depending on what preceded it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaubot.features._ta import sma, squash
from xaubot.features.base import EPS, FeatureContext, FeatureSpec, Transform


class PatternTransform(Transform):
    """Continuous strengths for the common price-action shapes."""

    group = "patterns"

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        self.cfg = config

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        cfg = self.cfg
        directional = (
            ("engulf", 2, "body engulfs the previous body in the opposite direction"),
            ("pinbar", 1, "long rejection wick with a small body"),
            ("rejection", 1, "large bar rejected from one extreme"),
            ("breakout", cfg.breakout_period + 1, "close beyond the prior N-bar range"),
            ("momentum", cfg.momentum_period + 1, "consecutive same-direction bodies"),
            ("double_retest", cfg.breakout_period + 1, "second test of a level with shallower penetration"),
        )
        out: list[FeatureSpec] = []
        for name, lookback, description in directional:
            out.append(
                FeatureSpec(f"{name}_bull_strength", self.group, lookback + 15, f"bullish {description}")
            )
            out.append(
                FeatureSpec(f"{name}_bear_strength", self.group, lookback + 15, f"bearish {description}")
            )
        out.append(
            FeatureSpec(
                "compression_strength",
                self.group,
                cfg.compression_period + 1,
                "how compressed this bar's range is against recent bars, [0,1]",
            )
        )
        return tuple(out)

    def _compute(self, bars: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        cfg = self.cfg
        open_, high, low, close = bars["open"], bars["high"], bars["low"], bars["close"]
        atr = ctx.atr
        bar_range = high - low
        body = close - open_
        abs_body = body.abs()
        out = pd.DataFrame(index=bars.index)

        # -- engulfing --------------------------------------------------------
        prev_body = body.shift(1)
        prev_open = open_.shift(1)
        prev_close = close.shift(1)
        body_ratio = abs_body / (prev_body.abs() + EPS)
        size = squash(abs_body / (atr + EPS), 1.0)

        bull_engulf = (body > 0) & (prev_body < 0) & (close > prev_open) & (open_ < prev_close)
        bear_engulf = (body < 0) & (prev_body > 0) & (close < prev_open) & (open_ > prev_close)
        engulf_quality = squash(body_ratio - 1.0, 1.0) * size
        out["engulf_bull_strength"] = engulf_quality.where(bull_engulf, 0.0)
        out["engulf_bear_strength"] = engulf_quality.where(bear_engulf, 0.0)

        # -- pin bar ----------------------------------------------------------
        upper_wick = (high - np.maximum(open_, close)) / (bar_range + EPS)
        lower_wick = (np.minimum(open_, close) - low) / (bar_range + EPS)
        small_body = 1.0 - (abs_body / (bar_range + EPS)).clip(0.0, 1.0)
        significant = squash(bar_range / (atr + EPS), 1.0)
        out["pinbar_bull_strength"] = (lower_wick * small_body * significant).clip(0.0, 1.0)
        out["pinbar_bear_strength"] = (upper_wick * small_body * significant).clip(0.0, 1.0)

        # -- rejection --------------------------------------------------------
        close_position = (close - low) / (bar_range + EPS)
        large = squash(bar_range / (atr + EPS), 1.5)
        out["rejection_bull_strength"] = (lower_wick * close_position * large).clip(0.0, 1.0)
        out["rejection_bear_strength"] = (upper_wick * (1.0 - close_position) * large).clip(0.0, 1.0)

        # -- breakout ---------------------------------------------------------
        period = cfg.breakout_period
        # shift(1) so the prior range excludes the breaking bar itself.
        prior_high = high.rolling(period, min_periods=period).max().shift(1)
        prior_low = low.rolling(period, min_periods=period).min().shift(1)
        expansion = squash(bar_range / (sma(bar_range, 20) + EPS) - 1.0, 1.0)
        out["breakout_bull_strength"] = (
            squash((close - prior_high) / (atr + EPS), 0.5) * (0.5 + 0.5 * expansion)
        ).where(close > prior_high, 0.0)
        out["breakout_bear_strength"] = (
            squash((prior_low - close) / (atr + EPS), 0.5) * (0.5 + 0.5 * expansion)
        ).where(close < prior_low, 0.0)

        # -- momentum ---------------------------------------------------------
        window = cfg.momentum_period
        up_body = body.clip(lower=0.0).rolling(window, min_periods=window).sum()
        down_body = (-body).clip(lower=0.0).rolling(window, min_periods=window).sum()
        agreement_up = (body > 0).rolling(window, min_periods=window).mean()
        out["momentum_bull_strength"] = squash(up_body / (window * atr + EPS), 0.3) * agreement_up
        out["momentum_bear_strength"] = squash(down_body / (window * atr + EPS), 0.3) * (1.0 - agreement_up)

        # -- compression ------------------------------------------------------
        comp = cfg.compression_period
        rank = bar_range.rolling(comp, min_periods=comp).rank(pct=True)
        out["compression_strength"] = (1.0 - rank).clip(0.0, 1.0)

        # -- double retest ----------------------------------------------------
        bull_retest, bear_retest = _double_retest(high, low, atr, period)
        out["double_retest_bull_strength"] = bull_retest
        out["double_retest_bear_strength"] = bear_retest

        return out.fillna(0.0).clip(0.0, 1.0)


def _double_retest(
    high: pd.Series, low: pd.Series, atr: pd.Series, period: int
) -> tuple[pd.Series, pd.Series]:
    """Score a second test of a level that penetrates less deeply than the first.

    An approximation, and labelled as one: a full implementation would track
    discrete level objects the way the FVG and order-block modules do. This
    compares the current bar's penetration of the rolling extreme against the
    deepest penetration earlier in the window, which captures the "held better
    the second time" property that makes the pattern interesting, at a fraction
    of the cost.

    Both the rolling extreme and the prior penetration are computed with a
    ``shift(1)``, so the current bar is never part of the level it is testing.
    """
    prior_low = low.rolling(period, min_periods=period).min().shift(1)
    prior_high = high.rolling(period, min_periods=period).max().shift(1)

    # How far the current bar pushed past the level (negative = held above).
    bull_penetration = (prior_low - low) / (atr + EPS)
    bear_penetration = (high - prior_high) / (atr + EPS)

    prior_bull_penetration = bull_penetration.shift(1).rolling(period, min_periods=1).max()
    prior_bear_penetration = bear_penetration.shift(1).rolling(period, min_periods=1).max()

    touched_bull = (low - prior_low).abs() / (atr + EPS) < 0.5
    touched_bear = (high - prior_high).abs() / (atr + EPS) < 0.5

    bull = squash(prior_bull_penetration - bull_penetration, 0.5).where(touched_bull, 0.0)
    bear = squash(prior_bear_penetration - bear_penetration, 0.5).where(touched_bear, 0.0)
    return bull.fillna(0.0), bear.fillna(0.0)
