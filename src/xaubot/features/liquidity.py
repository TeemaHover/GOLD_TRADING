"""Reference levels, distances, and liquidity sweeps.

Traders and algorithms cluster stops just beyond obvious levels -- yesterday's
high, the Asian session low, a row of equal highs. Price reaching for that
liquidity and then rejecting is one of the few genuinely repeatable structures
in intraday gold, so it gets first-class feature treatment.

**The point-in-time trap.** "Previous day's high" computed with a naive
``groupby(date).transform("max")`` gives every bar of a day the high of the day
it is *in*, including bars from the morning that could not possibly know the
afternoon's high. Every level here is instead built as a table of *completed*
periods stamped with the period's end time, then joined with ``merge_asof``
backwards -- the same mechanism as higher-timeframe context, for the same
reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaubot.core.enums import Session
from xaubot.core.time_utils import CLOSE_TIME
from xaubot.features._ta import squash
from xaubot.features.base import EPS, FeatureContext, FeatureSpec, Transform

#: Reference levels tracked for distance and sweep detection.
_LEVEL_NAMES = (
    "prev_day_high",
    "prev_day_low",
    "prev_week_high",
    "prev_week_low",
    "asia_high",
    "asia_low",
    "london_high",
    "london_low",
    "ny_high",
    "ny_low",
    "session_high",
    "session_low",
)


class LiquidityTransform(Transform):
    """Level distances, sweep detection, and stop-hunt scoring."""

    group = "liquidity"

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        self.cfg = config

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        cfg = self.cfg
        note = "levels come from completed periods only, joined as-of on close_time"
        out: list[FeatureSpec] = [
            FeatureSpec(
                f"dist_{name}_atr", self.group, 288, f"distance from close to {name}, ATR units", note
            )
            for name in _LEVEL_NAMES
        ]
        out += [
            FeatureSpec(
                "nearest_level_above_atr", self.group, 288, "distance to the closest level above", note
            ),
            FeatureSpec(
                "nearest_level_below_atr", self.group, 288, "distance to the closest level below", note
            ),
            FeatureSpec(
                "level_cluster_density",
                self.group,
                288,
                f"fraction of tracked levels within {cfg.level_cluster_atr} ATR of price",
                note,
            ),
            FeatureSpec(
                "sweep_high", self.group, 288, "wick pierced a level above and closed back below", note
            ),
            FeatureSpec(
                "sweep_low", self.group, 288, "wick pierced a level below and closed back above", note
            ),
            FeatureSpec(
                "sweep_strength",
                self.group,
                288,
                "pierce depth times close-back fraction, [0,1]",
                note,
            ),
            FeatureSpec(
                "sweep_age", self.group, 288, "bars since the most recent sweep in either direction", note
            ),
            FeatureSpec(
                "sweep_direction", self.group, 288, "-1 low swept, +1 high swept, 0 none recently", note
            ),
            FeatureSpec(
                "stop_hunt_score",
                self.group,
                288,
                "recent sweep followed by displacement the other way, [0,1]",
                "measured from the sweep bar forward to the current bar only",
            ),
            FeatureSpec(
                "rejection_after_sweep",
                self.group,
                288,
                "how decisively price rejected the swept level, [0,1]",
                "measured from the sweep bar forward to the current bar only",
            ),
            FeatureSpec(
                f"equal_highs_count_{cfg.equal_level_period}",
                self.group,
                cfg.equal_level_period,
                "bars whose high sits near the rolling high: a liquidity pool",
            ),
            FeatureSpec(
                f"equal_lows_count_{cfg.equal_level_period}",
                self.group,
                cfg.equal_level_period,
                "bars whose low sits near the rolling low: a liquidity pool",
            ),
        ]
        return tuple(out)

    def _compute(self, bars: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        cfg = self.cfg
        high, low, close = bars["high"], bars["low"], bars["close"]
        atr = ctx.atr
        out = pd.DataFrame(index=bars.index)

        levels = self._build_levels(bars, ctx)

        for name in _LEVEL_NAMES:
            out[f"dist_{name}_atr"] = (close - levels[name]) / (atr + EPS)

        level_matrix = levels.to_numpy()
        close_values = close.to_numpy()[:, None]
        atr_values = (atr.to_numpy() + EPS)[:, None]

        with np.errstate(invalid="ignore"):
            offsets = (level_matrix - close_values) / atr_values
            above = np.where(offsets > 0, offsets, np.nan)
            below = np.where(offsets < 0, -offsets, np.nan)

        out["nearest_level_above_atr"] = _nanmin(above)
        out["nearest_level_below_atr"] = _nanmin(below)
        with np.errstate(invalid="ignore"):
            near = np.abs(offsets) <= cfg.level_cluster_atr
        counted = np.isfinite(offsets).sum(axis=1)
        out["level_cluster_density"] = np.where(
            counted > 0, near.sum(axis=1) / np.maximum(counted, 1), np.nan
        )

        sweeps = self._detect_sweeps(bars, levels, ctx)
        out["sweep_high"] = sweeps["sweep_high"]
        out["sweep_low"] = sweeps["sweep_low"]
        out["sweep_strength"] = sweeps["sweep_strength"]
        out["sweep_age"] = sweeps["sweep_age"]
        out["sweep_direction"] = sweeps["sweep_direction"]
        out["stop_hunt_score"] = sweeps["stop_hunt_score"]
        out["rejection_after_sweep"] = sweeps["rejection_after_sweep"]

        period = cfg.equal_level_period
        tolerance = cfg.equal_level_tolerance_atr * atr
        rolling_high = high.rolling(period, min_periods=period).max()
        rolling_low = low.rolling(period, min_periods=period).min()
        out[f"equal_highs_count_{period}"] = (
            (high >= rolling_high - tolerance).rolling(period, min_periods=period).sum()
        )
        out[f"equal_lows_count_{period}"] = (
            (low <= rolling_low + tolerance).rolling(period, min_periods=period).sum()
        )

        return out

    # -- level construction ----------------------------------------------
    def _build_levels(self, bars: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        """Assemble every reference level, all point-in-time correct."""
        stamps = pd.DatetimeIndex(bars.index)
        levels = pd.DataFrame(index=bars.index)

        trading_day = pd.Index(ctx.calendar.trading_day(stamps))
        day_levels = _completed_period_levels(bars, trading_day)
        levels["prev_day_high"] = day_levels["high"]
        levels["prev_day_low"] = day_levels["low"]

        week = pd.Index(pd.DatetimeIndex(trading_day).to_period("W").astype(str))
        week_levels = _completed_period_levels(bars, week)
        levels["prev_week_high"] = week_levels["high"]
        levels["prev_week_low"] = week_levels["low"]

        sessions = ctx.sessions.reindex(bars.index)
        for session, column in (
            (Session.ASIA, "in_asia"),
            (Session.LONDON, "in_london"),
            (Session.NY, "in_ny"),
        ):
            prefix = session.value.lower()
            mask = sessions[column].to_numpy(dtype=bool)
            if mask.any():
                subset = bars.loc[mask]
                keys = sessions.loc[mask, "session_id"]
                session_levels = _completed_period_levels(subset, pd.Index(keys), target_index=bars.index)
                levels[f"{prefix}_high"] = session_levels["high"]
                levels[f"{prefix}_low"] = session_levels["low"]
            else:
                levels[f"{prefix}_high"] = np.nan
                levels[f"{prefix}_low"] = np.nan

        # Running extremes of the session in progress. Expanding within the
        # group is causal: bar i sees only bars of this session up to i.
        session_id = sessions["session_id"]
        levels["session_high"] = bars["high"].groupby(session_id).cummax()
        levels["session_low"] = bars["low"].groupby(session_id).cummin()

        return levels.loc[:, list(_LEVEL_NAMES)]

    # -- sweep detection --------------------------------------------------
    def _detect_sweeps(
        self, bars: pd.DataFrame, levels: pd.DataFrame, ctx: FeatureContext
    ) -> dict[str, np.ndarray]:
        """Detect liquidity sweeps and score what happened after them."""
        cfg = self.cfg
        atr = ctx.atr.to_numpy()
        high = bars["high"].to_numpy()
        low = bars["low"].to_numpy()
        close = bars["close"].to_numpy()

        # Only levels that existed *before* this bar can be swept by it. The
        # running session high/low are excluded: they are defined by this bar,
        # so "piercing" them is circular.
        sweepable = levels.drop(columns=["session_high", "session_low"]).shift(1).to_numpy()

        tolerance = cfg.sweep_tolerance_atr * atr

        with np.errstate(invalid="ignore"):
            pierced_up = (high[:, None] > sweepable + tolerance[:, None]) & (close[:, None] < sweepable)
            pierced_down = (low[:, None] < sweepable - tolerance[:, None]) & (close[:, None] > sweepable)

            depth_up = np.where(pierced_up, (high[:, None] - sweepable) / (atr[:, None] + EPS), np.nan)
            depth_down = np.where(pierced_down, (sweepable - low[:, None]) / (atr[:, None] + EPS), np.nan)

        swept_high = np.nan_to_num(pierced_up.sum(axis=1), nan=0.0) > 0
        swept_low = np.nan_to_num(pierced_down.sum(axis=1), nan=0.0) > 0

        max_depth = np.fmax(_nanmax(depth_up), _nanmax(depth_down))
        bar_range = high - low
        close_back = np.where(
            swept_high,
            (high - close) / (bar_range + EPS),
            np.where(swept_low, (close - low) / (bar_range + EPS), np.nan),
        )
        strength = np.nan_to_num(squash(pd.Series(max_depth), 1.0).to_numpy() * close_back, nan=0.0)

        return _score_after_sweep(
            swept_high=swept_high,
            swept_low=swept_low,
            strength=strength,
            close=close,
            atr=atr,
            lookback=cfg.sweep_lookback,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _nanmin(matrix: np.ndarray) -> np.ndarray:
    """Row-wise nanmin that returns NaN for all-NaN rows without warning."""
    empty = np.isnan(matrix).all(axis=1)
    out = np.full(matrix.shape[0], np.nan)
    if (~empty).any():
        out[~empty] = np.nanmin(matrix[~empty], axis=1)
    return out


def _nanmax(matrix: np.ndarray) -> np.ndarray:
    """Row-wise nanmax that returns NaN for all-NaN rows without warning."""
    empty = np.isnan(matrix).all(axis=1)
    out = np.full(matrix.shape[0], np.nan)
    if (~empty).any():
        out[~empty] = np.nanmax(matrix[~empty], axis=1)
    return out


def _completed_period_levels(
    bars: pd.DataFrame,
    group_key: pd.Index,
    target_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """High/low of the most recently *completed* period, as of each bar.

    Args:
        bars: Bars to aggregate.
        group_key: Period label per bar (trading day, week, session instance).
        target_index: Index to project onto. Defaults to ``bars.index``.

    Returns:
        Frame indexed by ``target_index`` with ``high``/``low`` of the latest
        period that had already finished at that timestamp.

    The mechanism is deliberately the same as higher-timeframe alignment: each
    period is stamped with the close time of its final bar, and a backward
    ``merge_asof`` guarantees a period is only visible once it has ended.
    """
    index = bars.index if target_index is None else target_index

    grouped = bars.groupby(group_key, sort=True)
    summary = pd.DataFrame(
        {
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "period_end": grouped.apply(lambda g: g.index[-1], include_groups=False),
        }
    ).sort_values("period_end")

    if summary.empty:
        return pd.DataFrame({"high": np.nan, "low": np.nan}, index=index)

    merged = pd.merge_asof(
        pd.DataFrame({CLOSE_TIME: pd.DatetimeIndex(index)}),
        summary.reset_index(drop=True).rename(columns={"period_end": CLOSE_TIME}),
        on=CLOSE_TIME,
        direction="backward",
        # Exact matches are DISALLOWED here, unlike the higher-timeframe join.
        # The difference matters and is not cosmetic:
        #
        #   - An HTF bar's close_time is a true boundary. A 15m bar closing at
        #     10:15 has closed at 10:15, so a 10:15 decision may read it.
        #   - A period's stamp is the close time of the last bar *inside* it.
        #     Matching that exactly would give the final bar of a day its own
        #     day's high as "previous day high" -- a value computed partly from
        #     the bar asking the question.
        #
        # It also fixes a live-only failure the replay audit caught: with
        # truncated history the in-progress period's last available bar looks
        # like the period end, so an exact match would expose the running high
        # of the day still in progress. The backtest, having future bars, would
        # not make that mistake -- so this would have diverged silently in
        # production only.
        allow_exact_matches=False,
    )
    merged.index = index
    return merged.loc[:, ["high", "low"]]


def _score_after_sweep(
    swept_high: np.ndarray,
    swept_low: np.ndarray,
    strength: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    lookback: int,
) -> dict[str, np.ndarray]:
    """Score the aftermath of a sweep, looking only backwards.

    A stop hunt is "liquidity was taken, then price went the other way". The
    naive implementation checks what price does over the *next* few bars, which
    is a direct look-ahead. Here the score is computed at the current bar from
    the displacement that has already happened since the sweep, so it starts at
    zero and builds as the move develops -- which is also how a trader would
    actually see it unfold.
    """
    n = len(close)
    out = {
        name: np.zeros(n)
        for name in ("sweep_high", "sweep_low", "sweep_strength", "stop_hunt_score", "rejection_after_sweep")
    }
    out["sweep_age"] = np.full(n, np.nan)
    out["sweep_direction"] = np.zeros(n)

    last_index = -1
    last_direction = 0.0
    last_close = np.nan
    last_strength = 0.0

    for i in range(n):
        out["sweep_high"][i] = 1.0 if swept_high[i] else 0.0
        out["sweep_low"][i] = 1.0 if swept_low[i] else 0.0
        out["sweep_strength"][i] = strength[i] if (swept_high[i] or swept_low[i]) else 0.0

        if swept_high[i] or swept_low[i]:
            last_index = i
            last_direction = 1.0 if swept_high[i] else -1.0
            last_close = close[i]
            last_strength = out["sweep_strength"][i]

        if last_index < 0:
            continue

        age = i - last_index
        out["sweep_age"][i] = age

        if age > lookback:
            out["sweep_direction"][i] = 0.0
            continue

        out["sweep_direction"][i] = last_direction
        # Displacement away from the swept side, in ATR units.
        displacement = -last_direction * (close[i] - last_close) / (atr[i] + EPS)
        follow_through = float(np.tanh(max(displacement, 0.0)))
        out["stop_hunt_score"][i] = last_strength * follow_through
        out["rejection_after_sweep"][i] = follow_through

    return out
