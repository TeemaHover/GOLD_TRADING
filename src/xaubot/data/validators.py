"""Data validation.

Validation *reports*, it does not repair -- repair belongs to
:mod:`xaubot.data.cleaning`. Splitting them means the quality report describes
the source as it actually arrived, which is the only way to notice that a feed
has quietly started emitting broken bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaubot.config.schema import ValidationConfig
from xaubot.core.calendar import TradingCalendar
from xaubot.core.enums import DataIssue, Severity, Timeframe
from xaubot.core.errors import SchemaError
from xaubot.core.logging import get_logger
from xaubot.core.time_utils import BAR_COLUMNS, CLOSE_TIME, OPEN_TIME, is_on_grid
from xaubot.core.types import GapReport, Issue

logger = get_logger(__name__)

_PRICE_COLUMNS = ("open", "high", "low", "close")


def validate_schema(frame: pd.DataFrame) -> None:
    """Raise if the canonical columns are absent or wrongly typed."""
    missing = [c for c in BAR_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(f"Frame missing canonical columns: {missing}")
    for column in (OPEN_TIME, CLOSE_TIME):
        if not pd.api.types.is_datetime64_any_dtype(frame[column]):
            raise SchemaError(f"{column} must be datetime64, got {frame[column].dtype}")
        if getattr(frame[column].dt, "tz", None) is None:
            raise SchemaError(f"{column} must be timezone-aware")
    for column in _PRICE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise SchemaError(f"{column} must be numeric, got {frame[column].dtype}")


def _examples(stamps: pd.Series | pd.DatetimeIndex, mask: np.ndarray, limit: int = 3) -> tuple[str, ...]:
    selected = pd.DatetimeIndex(pd.Series(stamps)[mask])[:limit]
    return tuple(str(s) for s in selected)


def detect_issues(
    frame: pd.DataFrame,
    timeframe: Timeframe,
    config: ValidationConfig,
) -> list[Issue]:
    """Enumerate every data-quality problem in an unrepaired frame.

    Args:
        frame: Canonical (but not yet cleaned) bar frame.
        timeframe: Expected bar timeframe, used for the grid check.
        config: Thresholds controlling what counts as a problem.

    Returns:
        A list of :class:`~xaubot.core.types.Issue`, empty if the data is clean.
    """
    validate_schema(frame)
    issues: list[Issue] = []
    stamps = pd.DatetimeIndex(frame[CLOSE_TIME])

    # -- ordering and uniqueness -----------------------------------------
    duplicated = np.asarray(stamps.duplicated(keep=False))
    if duplicated.any():
        issues.append(
            Issue(
                kind=DataIssue.DUPLICATE_TIMESTAMP,
                severity=Severity.WARNING,
                count=int(stamps.duplicated().sum()),
                detail="Repeated close timestamps; cleaning will apply duplicate_policy.",
                examples=_examples(stamps, duplicated),
            )
        )

    if not stamps.is_monotonic_increasing:
        backwards = np.concatenate([[False], stamps[1:].to_numpy() < stamps[:-1].to_numpy()])
        issues.append(
            Issue(
                kind=DataIssue.NON_MONOTONIC,
                severity=Severity.WARNING,
                count=int(backwards.sum()),
                detail="Rows are out of chronological order; cleaning will sort them.",
                examples=_examples(stamps, backwards),
            )
        )

    on_grid = is_on_grid(stamps, timeframe).to_numpy()
    if not on_grid.all():
        issues.append(
            Issue(
                kind=DataIssue.OFF_GRID_TIMESTAMP,
                severity=Severity.WARNING,
                count=int((~on_grid).sum()),
                detail=f"Timestamps not aligned to the {timeframe.value} grid.",
                examples=_examples(stamps, ~on_grid),
            )
        )

    # -- nulls ------------------------------------------------------------
    price_null = frame[list(_PRICE_COLUMNS)].isna().any(axis=1).to_numpy()
    if price_null.any():
        count = int(price_null.sum())
        fraction = count / len(frame)
        issues.append(
            Issue(
                kind=DataIssue.NULL_VALUE,
                severity=Severity.ERROR if fraction > config.max_null_fraction else Severity.WARNING,
                count=count,
                detail=f"{fraction:.4%} of rows have a null price (limit {config.max_null_fraction:.2%}).",
                examples=_examples(stamps, price_null),
            )
        )

    # -- price sanity -----------------------------------------------------
    non_positive = (frame[list(_PRICE_COLUMNS)] <= 0).any(axis=1).to_numpy()
    if non_positive.any():
        issues.append(
            Issue(
                kind=DataIssue.NON_POSITIVE_PRICE,
                severity=Severity.ERROR,
                count=int(non_positive.sum()),
                detail="Prices must be strictly positive.",
                examples=_examples(stamps, non_positive),
            )
        )

    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    open_ = frame["open"].to_numpy()
    close = frame["close"].to_numpy()
    with np.errstate(invalid="ignore"):
        inconsistent = (high < low) | (high < open_) | (high < close) | (low > open_) | (low > close)
    inconsistent = np.nan_to_num(inconsistent, nan=False).astype(bool)
    if inconsistent.any():
        issues.append(
            Issue(
                kind=DataIssue.OHLC_INCONSISTENT,
                severity=Severity.ERROR,
                count=int(inconsistent.sum()),
                detail="high < low, or open/close outside the high-low range.",
                examples=_examples(stamps, inconsistent),
            )
        )

    flat = np.nan_to_num(high == low, nan=False).astype(bool)
    if flat.any():
        issues.append(
            Issue(
                kind=DataIssue.FLAT_BAR,
                severity=Severity.INFO,
                count=int(flat.sum()),
                detail="Zero-range bars: usually a dead market, occasionally a stale feed.",
                examples=_examples(stamps, flat),
            )
        )

    # -- volume -----------------------------------------------------------
    volume = frame["volume"].to_numpy(dtype="float64", na_value=np.nan)
    negative_volume = np.nan_to_num(volume < 0, nan=False).astype(bool)
    if negative_volume.any():
        issues.append(
            Issue(
                kind=DataIssue.NEGATIVE_VOLUME,
                severity=Severity.ERROR,
                count=int(negative_volume.sum()),
                detail="Negative volume is impossible; the feed is corrupt.",
                examples=_examples(stamps, negative_volume),
            )
        )
    if config.flag_zero_volume:
        zero_volume = np.nan_to_num(volume == 0, nan=False).astype(bool)
        if zero_volume.any():
            issues.append(
                Issue(
                    kind=DataIssue.ZERO_VOLUME,
                    severity=Severity.INFO,
                    count=int(zero_volume.sum()),
                    detail="Zero-volume bars; check whether the feed synthesises quiet periods.",
                    examples=_examples(stamps, zero_volume),
                )
            )

    # -- extreme returns --------------------------------------------------
    extreme = _extreme_return_mask(frame["close"], config.extreme_return_sigma)
    if extreme.any():
        issues.append(
            Issue(
                kind=DataIssue.EXTREME_RETURN,
                severity=Severity.WARNING,
                count=int(extreme.sum()),
                detail=(
                    f"Bar-to-bar returns beyond {config.extreme_return_sigma:g} robust sigma. "
                    "Could be a real news spike or a bad print - inspect before training."
                ),
                examples=_examples(stamps, extreme),
            )
        )

    return issues


def _extreme_return_mask(close: pd.Series, sigma_threshold: float) -> np.ndarray:
    """Flag returns beyond N robust standard deviations.

    Uses median absolute deviation rather than the sample standard deviation:
    a single bad print inflates the ordinary sigma enough to hide itself.
    """
    returns = np.log(close.astype("float64")).diff()
    finite = returns[np.isfinite(returns)]
    if len(finite) < 100:
        return np.zeros(len(close), dtype=bool)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad <= 0:
        return np.zeros(len(close), dtype=bool)
    robust_sigma = 1.4826 * mad
    z = (returns - median).abs() / robust_sigma
    return np.nan_to_num(z.to_numpy() > sigma_threshold, nan=False).astype(bool)


def detect_gaps(
    close_times: pd.DatetimeIndex,
    timeframe: Timeframe,
    calendar: TradingCalendar,
    max_gaps_reported: int = 20,
) -> GapReport:
    """Compare present bars against the bars the trading calendar expects.

    Weekend and maintenance-break bars are excluded, so the reported
    completeness reflects feed quality rather than market hours.
    """
    if len(close_times) == 0:
        return GapReport(
            timeframe=timeframe,
            expected_bars=0,
            present_bars=0,
            missing_bars=0,
            largest_gap_bars=0,
            largest_gap_start=None,
        )

    duration = pd.Timedelta(timeframe.duration)
    open_times = close_times - duration
    expected = calendar.expected_bars(open_times.min(), open_times.max(), timeframe)

    present = pd.DatetimeIndex(open_times.unique())
    missing = expected.difference(present)

    largest_gap = 0
    largest_start: pd.Timestamp | None = None
    reported: list[tuple[str, str, int]] = []

    if len(missing):
        # Group consecutive missing timestamps into runs. Compared as Timedelta
        # rather than as raw integers: pandas datetime resolution varies between
        # ns and us depending on how the frame was built, and an integer compare
        # against Timedelta.value silently splits every run into single bars.
        deltas = pd.Series(missing).diff()
        breaks = np.concatenate([[True], (deltas.iloc[1:] != duration).to_numpy()])
        run_id = np.cumsum(breaks)
        runs = pd.Series(missing, index=run_id).groupby(level=0)
        summaries = [(pd.Timestamp(g.iloc[0]), pd.Timestamp(g.iloc[-1]), len(g)) for _, g in runs]
        summaries.sort(key=lambda item: item[2], reverse=True)
        largest_start, _, largest_gap = summaries[0]
        reported = [
            (str(start), str(end + duration), bars) for start, end, bars in summaries[:max_gaps_reported]
        ]

    return GapReport(
        timeframe=timeframe,
        expected_bars=len(expected),
        present_bars=int(present.isin(expected).sum()),
        missing_bars=len(missing),
        largest_gap_bars=int(largest_gap),
        largest_gap_start=largest_start,
        gaps_over_threshold=tuple(reported),
    )
