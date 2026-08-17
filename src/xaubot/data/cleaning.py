"""Data repair.

Repairs are conservative and logged. In particular, missing bars are **not**
forward-filled by default: a synthetic bar is a fabricated observation that the
model will learn from as if it were real, and a fabricated high/low can trigger
a stop in the backtester that never existed in the market. Leaving the gap is
honest, and the feature engine can mark bars that follow one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from xaubot.config.schema import CleaningConfig
from xaubot.core.enums import Timeframe
from xaubot.core.logging import get_logger
from xaubot.core.time_utils import BAR_COLUMNS, CLOSE_TIME, OPEN_TIME, is_on_grid

logger = get_logger(__name__)

_PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(slots=True)
class CleaningResult:
    """Cleaned frame plus an audit trail of what was removed."""

    frame: pd.DataFrame
    dropped: dict[str, int] = field(default_factory=dict)
    filled_bars: int = 0

    @property
    def total_dropped(self) -> int:
        return sum(self.dropped.values())


def clean_bars(
    frame: pd.DataFrame,
    timeframe: Timeframe,
    config: CleaningConfig,
) -> CleaningResult:
    """Sort, deduplicate, and drop unusable rows.

    Args:
        frame: Canonical but unrepaired bar frame (from a loader).
        timeframe: Bar timeframe, used for the grid check and gap filling.
        config: Cleaning policy.

    Returns:
        A :class:`CleaningResult` whose ``frame`` is indexed by close time and
        satisfies the :class:`~xaubot.core.types.BarFrame` invariants.
    """
    working = frame.copy()
    dropped: dict[str, int] = {}
    before = len(working)

    def _drop(mask: np.ndarray, reason: str) -> None:
        nonlocal working
        count = int(mask.sum())
        if count:
            dropped[reason] = dropped.get(reason, 0) + count
            working = working.loc[~mask].copy()
            logger.warning("Dropped %d rows: %s", count, reason)

    if config.drop_null_rows:
        _drop(working[list(_PRICE_COLUMNS)].isna().any(axis=1).to_numpy(), "null_price")

    if config.drop_non_positive_prices and len(working):
        _drop((working[list(_PRICE_COLUMNS)] <= 0).any(axis=1).to_numpy(), "non_positive_price")

    if len(working):
        high = working["high"].to_numpy()
        low = working["low"].to_numpy()
        open_ = working["open"].to_numpy()
        close = working["close"].to_numpy()
        with np.errstate(invalid="ignore"):
            inconsistent = (high < low) | (high < open_) | (high < close) | (low > open_) | (low > close)
        inconsistent = np.nan_to_num(inconsistent, nan=False).astype(bool)

        if config.clip_ohlc_inconsistent and inconsistent.any():
            # Repair by widening the bar to contain open/close. Only ever safe
            # for feeds with known rounding artefacts, so it is opt-in.
            working.loc[inconsistent, "high"] = np.maximum.reduce(
                [
                    working.loc[inconsistent, "high"].to_numpy(),
                    working.loc[inconsistent, "open"].to_numpy(),
                    working.loc[inconsistent, "close"].to_numpy(),
                ]
            )
            working.loc[inconsistent, "low"] = np.minimum.reduce(
                [
                    working.loc[inconsistent, "low"].to_numpy(),
                    working.loc[inconsistent, "open"].to_numpy(),
                    working.loc[inconsistent, "close"].to_numpy(),
                ]
            )
            logger.warning("Clipped %d OHLC-inconsistent bars", int(inconsistent.sum()))
        elif config.drop_ohlc_inconsistent:
            _drop(inconsistent, "ohlc_inconsistent")

    if config.drop_off_grid and len(working):
        stamps = pd.DatetimeIndex(working[CLOSE_TIME])
        _drop(~is_on_grid(stamps, timeframe).to_numpy(), "off_grid_timestamp")

    if len(working):
        # Stable sort keeps the original file order among equal timestamps, so
        # keep_first/keep_last mean what they say.
        working = working.sort_values(CLOSE_TIME, kind="mergesort")

        duplicated_mask = working[CLOSE_TIME].duplicated(keep=False).to_numpy()
        if duplicated_mask.any():
            if config.duplicate_policy == "raise":
                stamps = working.loc[duplicated_mask, CLOSE_TIME].head(5).tolist()
                raise ValueError(f"Duplicate timestamps present (policy='raise'): {stamps}")
            keep = "last" if config.duplicate_policy == "keep_last" else "first"
            drop_mask = working[CLOSE_TIME].duplicated(keep=keep).to_numpy()
            _drop(drop_mask, f"duplicate_timestamp({config.duplicate_policy})")

    working = working.loc[:, list(BAR_COLUMNS)]
    working[OPEN_TIME] = pd.DatetimeIndex(working[OPEN_TIME])
    working[CLOSE_TIME] = pd.DatetimeIndex(working[CLOSE_TIME])
    working = working.set_index(CLOSE_TIME, drop=False)
    working.index.name = CLOSE_TIME

    filled = 0
    if config.fill_missing_bars and config.max_forward_fill_bars > 0:
        working, filled = _fill_small_gaps(working, timeframe, config.max_forward_fill_bars)

    logger.info(
        "Cleaning: %d rows in, %d out (%d dropped, %d synthesised)",
        before,
        len(working),
        before - len(working) + filled,
        filled,
    )
    return CleaningResult(frame=working, dropped=dropped, filled_bars=filled)


def _fill_small_gaps(frame: pd.DataFrame, timeframe: Timeframe, max_run: int) -> tuple[pd.DataFrame, int]:
    """Forward-fill runs of at most ``max_run`` missing bars.

    Synthetic bars are zero-range (``open == high == low == close``) with zero
    volume, so they are visibly inert rather than pretending to be real price
    action, and carry ``is_synthetic=True`` for downstream masking.
    """
    duration = pd.Timedelta(timeframe.duration)
    index = pd.DatetimeIndex(frame.index)
    deltas = index[1:] - index[:-1]
    gap_positions = np.flatnonzero(deltas > duration)
    if len(gap_positions) == 0:
        frame["is_synthetic"] = False
        return frame, 0

    additions: list[pd.DataFrame] = []
    for pos in gap_positions:
        missing_count = int(deltas[pos] / duration) - 1
        if missing_count > max_run:
            continue
        start = index[pos] + duration
        stamps = pd.date_range(start=start, periods=missing_count, freq=timeframe.pandas_freq, tz="UTC")
        last_close = float(frame["close"].iloc[pos])
        additions.append(
            pd.DataFrame(
                {
                    OPEN_TIME: stamps - duration,
                    CLOSE_TIME: stamps,
                    "open": last_close,
                    "high": last_close,
                    "low": last_close,
                    "close": last_close,
                    "volume": 0.0,
                    "is_synthetic": True,
                },
                index=stamps,
            )
        )

    frame["is_synthetic"] = False
    if not additions:
        return frame, 0

    filled = pd.concat([frame, *additions]).sort_index()
    filled.index.name = CLOSE_TIME
    count = len(filled) - len(frame)
    logger.warning("Synthesised %d bars to fill gaps of <= %d bars", count, max_run)
    return filled, count
