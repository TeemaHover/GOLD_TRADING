"""Higher-timeframe construction.

This module is the second half of the point-in-time guarantee (the first being
the timestamp contract in :mod:`xaubot.core.time_utils`). Two rules govern it:

1. **Bars are grouped by open time and stamped with their close time.** A 15m
   bar covering 10:00-10:15 carries ``close_time = 10:15``. Downstream,
   ``merge_asof`` on ``close_time`` with ``direction="backward"`` then makes it
   impossible for a 5m decision at 10:15 to see the 10:15-10:30 bar.

2. **The trailing incomplete bar is dropped.** It would otherwise be stored as
   if it were a finished bar, and any consumer that ignored ``close_time``
   would read the future.

Anchoring is configurable because it is not arbitrary: gold's daily bar closes
at 17:00 New York, and a 4h grid anchored to the wrong hour produces context
bars that disagree with every chart the trader is looking at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaubot.config.schema import ResampleConfig
from xaubot.core.calendar import TradingCalendar
from xaubot.core.enums import Timeframe
from xaubot.core.errors import ResampleError
from xaubot.core.logging import get_logger
from xaubot.core.time_utils import CLOSE_TIME, OPEN_TIME, floor_to_timeframe
from xaubot.core.types import BarFrame

logger = get_logger(__name__)

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

_AGGREGATION = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def _hhmm_minutes(value: str) -> int:
    hours, _, minutes = value.partition(":")
    return int(hours) * 60 + int(minutes or 0)


def _daily_bounds(
    open_times: pd.DatetimeIndex, tz: str, anchor_time: str
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Map bar open times to the open/close of their anchored trading day.

    Anchoring is done in local wall-clock terms rather than by adding a fixed
    UTC offset, so the boundary stays at 17:00 New York year-round instead of
    drifting by an hour twice a year.
    """
    anchor_minutes = _hhmm_minutes(anchor_time)
    local = open_times.tz_convert(tz)

    # Shift back by the anchor so that "which trading day is this?" becomes a
    # plain date question, then recover the boundary instants in local time.
    shifted_naive = (local - pd.Timedelta(minutes=anchor_minutes)).tz_localize(None)
    day = pd.DatetimeIndex(shifted_naive).normalize()

    anchor_offset = pd.Timedelta(minutes=anchor_minutes)
    # 17:00 local never lands on a DST transition (those happen around 02:00),
    # so localisation here is unambiguous by construction.
    start_local = (day + anchor_offset).tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
    end_local = (day + pd.Timedelta(days=1) + anchor_offset).tz_localize(
        tz, ambiguous=True, nonexistent="shift_forward"
    )
    return start_local.tz_convert("UTC"), end_local.tz_convert("UTC")


def _group_key(
    open_times: pd.DatetimeIndex, target: Timeframe, config: ResampleConfig
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Return ``(group_open, group_close)`` for each base bar."""
    if target is Timeframe.D1:
        return _daily_bounds(open_times, config.daily_anchor_tz, config.daily_anchor_time)

    offset = pd.Timedelta(minutes=config.h4_anchor_offset_minutes if target is Timeframe.H4 else 0)
    origin = _EPOCH + offset
    group_open = floor_to_timeframe(open_times, target, origin=origin)
    return group_open, group_open + pd.Timedelta(target.duration)


def resample_bars(
    base: BarFrame,
    target: Timeframe,
    config: ResampleConfig | None = None,
    calendar: TradingCalendar | None = None,
) -> BarFrame:
    """Aggregate base bars into a coarser timeframe.

    Args:
        base: Source bars (typically 5m).
        target: Desired timeframe; must be a coarser multiple of ``base``.
        config: Anchoring and completeness policy.
        calendar: Optional trading calendar. When supplied, each output bar
            records how complete it is relative to the bars the calendar
            expected, which is what makes holiday half-days distinguishable
            from feed outages.

    Returns:
        A :class:`~xaubot.core.types.BarFrame` at ``target`` resolution with
        two extra diagnostic columns: ``n_base_bars`` and ``is_partial``.

    Raises:
        ResampleError: If the target is not a coarser multiple of the base.
    """
    config = config or ResampleConfig()

    if target.minutes <= base.timeframe.minutes:
        raise ResampleError(f"Cannot resample {base.timeframe} -> {target}: upsampling fabricates data")
    if target is not Timeframe.D1 and target.minutes % base.timeframe.minutes != 0:
        raise ResampleError(f"{target} is not an integer multiple of {base.timeframe}")
    if len(base) == 0:
        raise ResampleError("Cannot resample an empty BarFrame")

    source = base.df
    open_times = pd.DatetimeIndex(source[OPEN_TIME])
    group_open, group_close = _group_key(open_times, target, config)

    work = source.loc[:, ["open", "high", "low", "close", "volume"]].copy()
    work["_group_open"] = group_open
    work["_group_close"] = group_close

    grouped = work.groupby("_group_open", sort=True)
    aggregated = grouped.agg(_AGGREGATION)
    aggregated["n_base_bars"] = grouped.size()
    aggregated[CLOSE_TIME] = grouped["_group_close"].first()
    aggregated.index.name = OPEN_TIME
    aggregated = aggregated.reset_index()

    # -- completeness -----------------------------------------------------
    if calendar is not None:
        expected = np.array(
            [
                len(calendar.expected_bars(o, c - pd.Timedelta(base.timeframe.duration), base.timeframe))
                for o, c in zip(aggregated[OPEN_TIME], aggregated[CLOSE_TIME], strict=True)
            ],
            dtype="float64",
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            completeness = np.where(expected > 0, aggregated["n_base_bars"] / expected, np.nan)
    else:
        modal = float(aggregated["n_base_bars"].mode().iloc[0])
        completeness = aggregated["n_base_bars"].to_numpy() / modal

    aggregated["is_partial"] = np.nan_to_num(completeness, nan=1.0) < config.min_bars_for_complete

    # -- drop the trailing incomplete bar ---------------------------------
    if config.drop_incomplete_last_bar and len(aggregated):
        last_close = pd.Timestamp(aggregated[CLOSE_TIME].iloc[-1])
        if last_close > base.end:
            logger.info(
                "Dropping trailing incomplete %s bar (closes %s, data ends %s)",
                target.value,
                last_close,
                base.end,
            )
            aggregated = aggregated.iloc[:-1]

    if aggregated.empty:
        raise ResampleError(
            f"Resampling {base.timeframe} -> {target} produced no complete bars; "
            f"the source spans only {base.end - base.start}"
        )

    aggregated = aggregated.set_index(CLOSE_TIME, drop=False)
    aggregated.index.name = CLOSE_TIME
    ordered = aggregated.loc[
        :, [OPEN_TIME, CLOSE_TIME, "open", "high", "low", "close", "volume", "n_base_bars", "is_partial"]
    ]

    partial = int(ordered["is_partial"].sum())
    logger.info(
        "Resampled %d %s bars -> %d %s bars (%d flagged partial)",
        len(base),
        base.timeframe.value,
        len(ordered),
        target.value,
        partial,
    )
    return BarFrame(df=ordered, timeframe=target, symbol=base.symbol)


def resample_many(
    base: BarFrame,
    targets: tuple[Timeframe, ...],
    config: ResampleConfig | None = None,
    calendar: TradingCalendar | None = None,
) -> dict[Timeframe, BarFrame]:
    """Resample to several timeframes, returning a mapping keyed by timeframe."""
    return {tf: resample_bars(base, tf, config, calendar) for tf in targets}
