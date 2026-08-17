"""Higher-timeframe context alignment.

This module is where the single most dangerous leak in the system is prevented,
so the rule is worth restating: **a 5m decision at 10:15 may use the 15m bar
that closes at 10:15, and may not use the one that closes at 10:30.**

Two distinct kinds of higher-timeframe information are produced, and keeping
them separate is deliberate:

- **Completed-bar features** (``*_1h``, ``*_4h``, ...) come from finished HTF
  bars, joined with a backward ``merge_asof`` on ``close_time``.
- **Running features** (``running_*``) describe the HTF bar currently *in
  progress* -- how far into the 4h candle we are, where price sits in its range
  so far. These are legitimate and useful, but they must be built by
  aggregating completed 5m bars, never by reading the HTF frame, which would
  hand back the finished bar's high and low. The ``running_`` prefix keeps the
  distinction visible in the manifest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaubot.core.enums import Timeframe
from xaubot.core.errors import LeakageError
from xaubot.core.logging import get_logger
from xaubot.core.time_utils import CLOSE_TIME, OPEN_TIME, floor_to_timeframe
from xaubot.features.base import EPS

logger = get_logger(__name__)


def align_htf_features(
    base_index: pd.DatetimeIndex,
    htf_features: dict[Timeframe, pd.DataFrame],
    *,
    verify: bool = True,
) -> pd.DataFrame:
    """Join higher-timeframe features onto the base timeframe's index.

    Args:
        base_index: Base (5m) bar close times -- i.e. decision times.
        htf_features: Feature frames per timeframe, each indexed by that
            timeframe's bar close times.
        verify: Assert afterwards that no row borrowed a bar that had not
            closed. Cheap, and this is the check worth paying for.

    Returns:
        Frame indexed by ``base_index`` with columns suffixed by timeframe.

    Raises:
        LeakageError: If verification finds a row whose matched HTF bar closes
            after the decision time.
    """
    result = pd.DataFrame(index=base_index)
    left = pd.DataFrame({CLOSE_TIME: pd.DatetimeIndex(base_index)})

    for timeframe, frame in htf_features.items():
        suffix = f"_{timeframe.value}"
        right = frame.copy()
        right.columns = [f"{column}{suffix}" for column in right.columns]
        # reset_index first: the frame is indexed by close_time, so adding a
        # close_time column before dropping the index makes the name ambiguous.
        close_times = pd.DatetimeIndex(frame.index)
        right = right.reset_index(drop=True)
        right[CLOSE_TIME] = close_times
        right = right.sort_values(CLOSE_TIME).reset_index(drop=True)

        matched_column = f"__matched_close{suffix}"
        right[matched_column] = right[CLOSE_TIME]

        merged = pd.merge_asof(
            left,
            right,
            on=CLOSE_TIME,
            direction="backward",
            # A bar closing exactly at the decision time HAS closed, so it is
            # usable. This is the 15m-bar-at-10:15 case from the design doc.
            allow_exact_matches=True,
        )
        merged.index = base_index

        if verify:
            _verify_no_future_match(merged, matched_column, timeframe)

        result = result.join(merged.drop(columns=[CLOSE_TIME, matched_column]))

    return result


def _verify_no_future_match(merged: pd.DataFrame, matched_column: str, timeframe: Timeframe) -> None:
    """Fail loudly if any row matched a bar that had not closed yet."""
    matched = merged[matched_column]
    present = matched.notna()
    if not present.any():
        logger.warning("No %s context matched any base bar; check the date ranges", timeframe.value)
        return

    violations = matched[present] > pd.Series(merged.index, index=merged.index)[present]
    if violations.any():
        first = merged.index[present][violations][0]
        raise LeakageError(
            f"{int(violations.sum())} bars matched a {timeframe.value} bar that had not closed "
            f"(first at {first}). The as-of join is broken; every downstream result is invalid."
        )


def running_htf_features(
    bars: pd.DataFrame,
    timeframes: tuple[Timeframe, ...],
    atr: pd.Series,
    base_timeframe: Timeframe,
) -> pd.DataFrame:
    """Describe the higher-timeframe bar currently in progress.

    Built by aggregating *completed* base bars within the current HTF interval.
    Reading these off the HTF frame instead would expose the finished bar's
    high, low, and close -- the classic look-ahead this whole module exists to
    prevent.

    Args:
        bars: Base timeframe OHLCV, indexed by close time.
        timeframes: Higher timeframes to describe.
        atr: Base-timeframe ATR, for normalisation.
        base_timeframe: The execution timeframe.

    Returns:
        Frame indexed like ``bars`` with ``running_*`` columns per timeframe.
    """
    out = pd.DataFrame(index=bars.index)
    open_times = pd.DatetimeIndex(bars[OPEN_TIME])

    for timeframe in timeframes:
        if timeframe is Timeframe.D1:
            # The daily anchor is configurable and tz-dependent; approximating
            # it with a fixed grid here would silently disagree with the
            # resampler, so daily progress is left to the completed-bar path.
            continue

        suffix = f"_{timeframe.value}"
        group = floor_to_timeframe(open_times, timeframe)
        group_series = pd.Series(group, index=bars.index)

        running_high = bars["high"].groupby(group_series).cummax()
        running_low = bars["low"].groupby(group_series).cummin()
        running_open = bars["open"].groupby(group_series).transform("first")
        elapsed = bars.groupby(group_series).cumcount() + 1

        bars_per_htf = timeframe.minutes // base_timeframe.minutes
        span = running_high - running_low

        out[f"running_progress{suffix}"] = (elapsed / bars_per_htf).clip(0.0, 1.0)
        out[f"running_range_atr{suffix}"] = span / (atr + EPS)
        out[f"running_position{suffix}"] = (bars["close"] - running_low) / (span + EPS)
        out[f"running_return_atr{suffix}"] = (bars["close"] - running_open) / (atr + EPS)

    return out


def cross_timeframe_features(
    aligned: pd.DataFrame,
    timeframes: tuple[Timeframe, ...],
    base_trend: pd.Series | None = None,
) -> pd.DataFrame:
    """Summarise agreement and volatility relationships across timeframes.

    Trend agreement across timeframes is one of the few pieces of context that
    plausibly matters for a 5m entry, so it gets an explicit summary feature
    rather than being left for the model to reconstruct from a dozen individual
    EMA-stack columns.
    """
    out = pd.DataFrame(index=aligned.index)

    stack_columns = [
        f"ema_stack_score_{tf.value}" for tf in timeframes if f"ema_stack_score_{tf.value}" in aligned
    ]
    if stack_columns:
        stacks = aligned.loc[:, stack_columns]
        out["htf_trend_agreement"] = np.sign(stacks).mean(axis=1)
        # Weight longer timeframes more: a 4h trend constrains a 5m entry more
        # than a 15m one does.
        weights = np.array([_timeframe_weight(column) for column in stack_columns], dtype="float64")
        out["htf_alignment_score"] = (stacks.to_numpy() * weights).sum(axis=1) / weights.sum()
        if base_trend is not None:
            out["htf_base_agreement"] = np.sign(stacks).mean(axis=1) * np.sign(base_trend)

    for timeframe in timeframes:
        column = f"atr_pct_{timeframe.value}"
        if column in aligned and "atr_pct" in aligned:
            out[f"tf_vol_ratio_base_{timeframe.value}"] = aligned["atr_pct"] / (aligned[column] + EPS)

    return out


def _timeframe_weight(column: str) -> float:
    """Weight for a timeframe suffix: longer timeframes count for more."""
    suffix = column.rsplit("_", 1)[-1]
    try:
        return float(Timeframe(suffix).minutes) ** 0.5
    except ValueError:
        return 1.0
