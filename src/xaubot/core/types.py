"""Core domain types.

These are the objects that cross module boundaries. Keeping them here (rather
than letting each layer define its own dict shape) is what allows the feature
engine, the backtester, and the live runner to share one code path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Self

import pandas as pd

from xaubot.core.enums import DataIssue, Severity, Timeframe
from xaubot.core.errors import SchemaError
from xaubot.core.time_utils import BAR_COLUMNS, CLOSE_TIME, OHLCV, OPEN_TIME


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Contract specification for a tradable instrument.

    Position sizing is arithmetic on these numbers; getting ``tick_value`` or
    ``contract_size`` wrong silently scales every reported P&L, so they are
    explicit config rather than defaults buried in code.
    """

    symbol: str
    contract_size: float = 100.0  # ounces per 1.00 lot
    tick_size: float = 0.01
    tick_value: float = 1.0  # account currency per tick per 1.00 lot
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 50.0
    commission_per_lot: float = 7.0  # round turn, account currency
    typical_spread: float = 0.20  # price units
    quote_currency: str = "USD"

    def __post_init__(self) -> None:
        for name in ("contract_size", "tick_size", "tick_value", "min_lot", "lot_step", "max_lot"):
            if getattr(self, name) <= 0:
                raise ValueError(f"InstrumentSpec.{name} must be positive")
        if self.min_lot > self.max_lot:
            raise ValueError("min_lot exceeds max_lot")

    def price_to_ticks(self, price_distance: float) -> float:
        """Convert a price distance to a number of ticks."""
        return abs(price_distance) / self.tick_size

    def value_per_lot(self, price_distance: float) -> float:
        """Account-currency value of a price move, per 1.00 lot."""
        return self.price_to_ticks(price_distance) * self.tick_value

    def round_lots(self, lots: float) -> float:
        """Round a lot size down to a tradable increment, clamped to limits."""
        if lots < self.min_lot:
            return 0.0
        stepped = math.floor(lots / self.lot_step) * self.lot_step
        stepped = min(stepped, self.max_lot)
        # Guard against binary-float dust like 0.13000000000000003.
        decimals = max(0, -math.floor(math.log10(self.lot_step)))
        return round(stepped, decimals)


@dataclass(slots=True)
class BarFrame:
    """A validated OHLCV table for one symbol at one timeframe.

    Invariants enforced on construction:

    - index is a unique, monotonically increasing, tz-aware UTC index of
      **close** times;
    - ``open_time`` and ``close_time`` are both present and consistent with the
      timeframe duration;
    - the OHLCV columns exist and are float/int typed.

    The index is close time (not open time) because close time *is* decision
    time. Indexing on it makes ``df.loc[:t]`` trivially point-in-time correct.
    """

    df: pd.DataFrame
    timeframe: Timeframe
    symbol: str
    validated: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self._validate()
        self.validated = True

    def _validate(self) -> None:
        missing = [c for c in BAR_COLUMNS if c not in self.df.columns]
        if missing:
            raise SchemaError(f"BarFrame missing required columns: {missing}")

        idx = self.df.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise SchemaError(f"BarFrame index must be a DatetimeIndex, got {type(idx).__name__}")
        if idx.tz is None or str(idx.tz) != "UTC":
            raise SchemaError(f"BarFrame index must be tz-aware UTC, got tz={idx.tz}")
        if not idx.is_monotonic_increasing:
            raise SchemaError("BarFrame index must be monotonically increasing")
        if idx.has_duplicates:
            dupes = int(idx.duplicated().sum())
            raise SchemaError(f"BarFrame index has {dupes} duplicate timestamps")
        if not idx.equals(pd.DatetimeIndex(self.df[CLOSE_TIME])):
            raise SchemaError("BarFrame index must equal the close_time column")

        expected = pd.Timedelta(self.timeframe.duration)
        actual = pd.DatetimeIndex(self.df[CLOSE_TIME]) - pd.DatetimeIndex(self.df[OPEN_TIME])
        if len(actual):
            if self.timeframe is Timeframe.D1:
                # Daily bars anchored to a local exchange close (17:00 New York)
                # are 23h or 25h long on the two DST transition days each year.
                # That is correct behaviour, not corruption: the bar still spans
                # exactly one trading day.
                allowed = {
                    pd.Timedelta(hours=23),
                    pd.Timedelta(hours=24),
                    pd.Timedelta(hours=25),
                }
                bad_mask = ~actual.isin(list(allowed))
            else:
                bad_mask = actual != expected
            if bad_mask.any():
                bad = int(bad_mask.sum())
                example = actual[bad_mask][0]
                raise SchemaError(
                    f"{bad} bars have an invalid duration for timeframe {self.timeframe} "
                    f"(expected {expected}, saw e.g. {example})"
                )

    # -- convenience ------------------------------------------------------
    @property
    def close_times(self) -> pd.DatetimeIndex:
        """Decision times for these bars."""
        return pd.DatetimeIndex(self.df.index)

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.df.index[0])

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.df.index[-1])

    def __len__(self) -> int:
        return len(self.df)

    def as_of(self, t: pd.Timestamp) -> Self:
        """Return a copy containing only bars closed at or before ``t``.

        This is the primary point-in-time accessor. Using it (rather than
        slicing the raw DataFrame) keeps the intent visible at call sites.
        """
        return type(self)(df=self.df.loc[:t].copy(), timeframe=self.timeframe, symbol=self.symbol)

    def ohlcv(self) -> pd.DataFrame:
        """Just the OHLCV columns, indexed by close time."""
        return self.df.loc[:, list(OHLCV)]


@dataclass(frozen=True, slots=True)
class Issue:
    """One data-quality finding."""

    kind: DataIssue
    severity: Severity
    count: int
    detail: str = ""
    examples: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "count": self.count,
            "detail": self.detail,
            "examples": list(self.examples),
        }


@dataclass(frozen=True, slots=True)
class GapReport:
    """Missing-bar analysis against the trading calendar."""

    timeframe: Timeframe
    expected_bars: int
    present_bars: int
    missing_bars: int
    largest_gap_bars: int
    largest_gap_start: pd.Timestamp | None
    gaps_over_threshold: tuple[tuple[str, str, int], ...] = ()

    @property
    def completeness(self) -> float:
        """Fraction of expected bars actually present, in ``[0, 1]``."""
        return self.present_bars / self.expected_bars if self.expected_bars else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe.value,
            "expected_bars": self.expected_bars,
            "present_bars": self.present_bars,
            "missing_bars": self.missing_bars,
            "completeness": round(self.completeness, 6),
            "largest_gap_bars": self.largest_gap_bars,
            "largest_gap_start": (
                self.largest_gap_start.isoformat() if self.largest_gap_start is not None else None
            ),
            "gaps_over_threshold": [
                {"start": s, "end": e, "bars": n} for s, e, n in self.gaps_over_threshold
            ],
        }


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Complete data-quality assessment for one ingested dataset."""

    symbol: str
    timeframe: Timeframe
    source: str
    rows_in: int
    rows_out: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    issues: tuple[Issue, ...]
    gaps: GapReport | None
    source_sha256: str = ""

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out

    @property
    def has_errors(self) -> bool:
        return any(i.severity is Severity.ERROR for i in self.issues)

    def issue_counts(self) -> dict[str, int]:
        return {i.kind.value: i.count for i in self.issues}

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_dropped": self.rows_dropped,
            "start": self.start.isoformat() if self.start is not None else None,
            "end": self.end.isoformat() if self.end is not None else None,
            "has_errors": self.has_errors,
            "issues": [i.to_dict() for i in self.issues],
            "gaps": self.gaps.to_dict() if self.gaps is not None else None,
        }
