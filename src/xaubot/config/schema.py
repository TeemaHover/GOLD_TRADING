"""Configuration schema.

Every runtime knob in the system is declared here as a pydantic model. Three
consequences, all deliberate:

1. Unknown keys are **errors**, not warnings -- a typo in a YAML file fails at
   startup instead of silently leaving a default in place.
2. The resolved config is hashable, so a feature set or dataset can be
   content-addressed by the parameters that produced it.
3. Secrets never appear here. They are read from the environment only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from xaubot.core.enums import Session, Timeframe, TimestampConvention

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Positive = Annotated[float, Field(gt=0.0)]


class StrictModel(BaseModel):
    """Base model: forbid unknown keys, freeze after construction."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
class PathsConfig(StrictModel):
    """Filesystem layout. Overridable via ``XAUBOT_DATA_ROOT`` etc."""

    data_root: Path = Path("data")
    artifact_root: Path = Path("artifacts")
    raw_dir: str = "raw"
    canonical_dir: str = "canonical"
    resampled_dir: str = "resampled"
    features_dir: str = "features"
    labels_dir: str = "labels"
    datasets_dir: str = "datasets"
    reports_dir: str = "reports"

    def raw(self) -> Path:
        return self.data_root / self.raw_dir

    def canonical(self) -> Path:
        return self.data_root / self.canonical_dir

    def resampled(self) -> Path:
        return self.data_root / self.resampled_dir

    def reports(self) -> Path:
        return self.data_root / self.reports_dir


# --------------------------------------------------------------------------
# Instrument / calendar / sessions
# --------------------------------------------------------------------------
class InstrumentConfig(StrictModel):
    """Contract specification. See :class:`~xaubot.core.types.InstrumentSpec`."""

    symbol: str = "XAUUSD"
    contract_size: Positive = 100.0
    tick_size: Positive = 0.01
    tick_value: Positive = 1.0
    min_lot: Positive = 0.01
    lot_step: Positive = 0.01
    max_lot: Positive = 50.0
    commission_per_lot: float = Field(default=7.0, ge=0.0)
    typical_spread: float = Field(default=0.20, ge=0.0)
    quote_currency: str = "USD"


class CalendarConfig(StrictModel):
    """Weekly trading schedule, expressed in ``tz`` local time."""

    tz: str = "America/New_York"
    week_open_weekday: int = Field(default=6, ge=0, le=6)
    week_open_time: str = "18:00"
    week_close_weekday: int = Field(default=4, ge=0, le=6)
    week_close_time: str = "17:00"
    daily_break_start: str = "17:00"
    daily_break_end: str = "18:00"
    holidays: tuple[str, ...] = ()


class SessionWindowConfig(StrictModel):
    """One session, defined in its own local timezone so DST is handled."""

    session: Session
    tz: str
    start: str
    end: str
    enabled: bool = True


class SessionsConfig(StrictModel):
    """Session definitions and which of them the bot is allowed to trade."""

    windows: tuple[SessionWindowConfig, ...] = (
        SessionWindowConfig(session=Session.ASIA, tz="Asia/Tokyo", start="09:00", end="18:00"),
        SessionWindowConfig(session=Session.LONDON, tz="Europe/London", start="08:00", end="16:30"),
        SessionWindowConfig(session=Session.NY, tz="America/New_York", start="08:00", end="17:00"),
    )
    tradable: tuple[Session, ...] = (
        Session.LONDON,
        Session.NY,
        Session.LONDON_NY_OVERLAP,
    )


# --------------------------------------------------------------------------
# Data source & ingestion
# --------------------------------------------------------------------------
class CsvSourceConfig(StrictModel):
    """How to read one CSV file.

    ``timestamp_convention`` is required rather than inferred: assuming the
    wrong end of the bar shifts every feature by one bar, which is a silent
    look-ahead bias that no downstream test can catch.
    """

    path: Path
    timeframe: Timeframe = Timeframe.M5
    timestamp_column: str = "datetime"
    date_column: str | None = None  # for split DATE/TIME files (MT5 exports)
    time_column: str | None = None
    timestamp_convention: TimestampConvention = TimestampConvention.OPEN
    source_tz: str | None = None  # None => timestamps are already UTC or tz-aware
    delimiter: str | None = None  # None => sniff
    decimal: str = "."
    column_map: dict[str, str] = Field(default_factory=dict)
    volume_column: str | None = None  # None => auto-detect volume/tickvol/tick_volume
    volume_semantics: Literal["tick_count", "contracts", "unknown"] = "tick_count"
    skiprows: int = 0

    @field_validator("path")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return Path(str(value)).expanduser()


class ValidationConfig(StrictModel):
    """Thresholds that decide whether a dataset is fit to use."""

    min_completeness: Fraction = 0.90
    max_null_fraction: Fraction = 0.01
    extreme_return_sigma: Positive = 12.0
    max_gap_bars_reported: int = Field(default=20, ge=1)
    flag_zero_volume: bool = True
    fail_on_error: bool = True


class CleaningConfig(StrictModel):
    """What to do about bad rows.

    Missing bars are **not** forward-filled by default. A synthetic bar is a
    lie the model will happily learn from; leaving a gap is honest and the
    feature engine can mark it.
    """

    duplicate_policy: Literal["keep_last", "keep_first", "raise"] = "keep_last"
    drop_non_positive_prices: bool = True
    drop_ohlc_inconsistent: bool = True
    clip_ohlc_inconsistent: bool = False
    drop_null_rows: bool = True
    fill_missing_bars: bool = False
    max_forward_fill_bars: int = Field(default=0, ge=0)
    drop_off_grid: bool = True


class ResampleConfig(StrictModel):
    """Higher-timeframe construction.

    Anchors matter: MT5 daily gold bars close at 17:00 New York, not at UTC
    midnight, and a 4h grid anchored to the wrong hour produces context bars
    that no chart the user looks at will agree with.
    """

    daily_anchor_tz: str = "America/New_York"
    daily_anchor_time: str = "17:00"
    h4_anchor_offset_minutes: int = Field(default=0, ge=0, lt=240)
    min_bars_for_complete: Fraction = 0.5
    drop_incomplete_last_bar: bool = True


class DataConfig(StrictModel):
    """Everything about getting bars into the system."""

    symbol: str = "XAUUSD"
    base_timeframe: Timeframe = Timeframe.M5
    context_timeframes: tuple[Timeframe, ...] = (
        Timeframe.M15,
        Timeframe.H1,
        Timeframe.H4,
        Timeframe.D1,
    )
    source: CsvSourceConfig
    validation: ValidationConfig = ValidationConfig()
    cleaning: CleaningConfig = CleaningConfig()
    resample: ResampleConfig = ResampleConfig()

    @model_validator(mode="after")
    def _check_timeframes(self) -> DataConfig:
        base = self.base_timeframe.minutes
        for tf in self.context_timeframes:
            if tf.minutes <= base:
                raise ValueError(
                    f"context timeframe {tf} is not coarser than base {self.base_timeframe}; "
                    "upsampling would fabricate data"
                )
            if tf.minutes % base != 0:
                raise ValueError(f"context timeframe {tf} is not a multiple of {self.base_timeframe}")
        if self.source.timeframe != self.base_timeframe:
            raise ValueError(
                f"source timeframe {self.source.timeframe} != base timeframe {self.base_timeframe}"
            )
        return self


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
class PriceFeaturesConfig(StrictModel):
    """Candle geometry and multi-horizon returns."""

    enabled: bool = True
    return_periods: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50)
    zscore_periods: tuple[int, ...] = (20, 50, 100)


class VolatilityFeaturesConfig(StrictModel):
    """ATR, realised volatility, range expansion, regime buckets."""

    enabled: bool = True
    atr_periods: tuple[int, ...] = (14, 50)
    primary_atr_period: int = Field(default=14, gt=1)
    percentile_windows: tuple[int, ...] = (252, 1000)
    realized_vol_periods: tuple[int, ...] = (20, 50)
    bb_period: int = Field(default=20, gt=1)
    bb_percentile_window: int = Field(default=252, gt=10)
    regime_low_pct: Fraction = 0.20
    regime_high_pct: Fraction = 0.80
    regime_extreme_pct: Fraction = 0.95


class VolumeFeaturesConfig(StrictModel):
    """Volume/activity features.

    Disable wholesale when validating across feeds: MT5 "volume" on XAUUSD is a
    broker-specific tick count, so these features do not transfer.
    """

    enabled: bool = True
    ma_periods: tuple[int, ...] = (20, 50)
    zscore_period: int = Field(default=50, gt=1)
    time_of_day_days: int = Field(default=20, gt=1)


class TrendFeaturesConfig(StrictModel):
    """EMA stack, ADX/DI, oscillators, regression slope."""

    enabled: bool = True
    ema_periods: tuple[int, ...] = (20, 50, 100, 200)
    slope_lookback: int = Field(default=5, gt=0)
    adx_period: int = Field(default=14, gt=1)
    rsi_period: int = Field(default=14, gt=1)
    linreg_periods: tuple[int, ...] = (20, 50)
    efficiency_period: int = Field(default=20, gt=1)
    hurst_period: int = Field(default=100, gt=10)


class StructureFeaturesConfig(StrictModel):
    """Swings, break of structure, market-structure shift."""

    enabled: bool = True
    swing_k: int = Field(default=2, ge=1, le=10)
    structure_memory: int = Field(default=6, gt=1)
    range_period: int = Field(default=50, gt=1)


class LiquidityFeaturesConfig(StrictModel):
    """Reference levels, distances, sweeps, stop hunts."""

    enabled: bool = True
    sweep_lookback: int = Field(default=12, gt=0)
    sweep_tolerance_atr: float = Field(default=0.10, ge=0.0)
    level_cluster_atr: float = Field(default=0.50, gt=0.0)
    equal_level_period: int = Field(default=50, gt=1)
    equal_level_tolerance_atr: float = Field(default=0.10, gt=0.0)


class FvgFeaturesConfig(StrictModel):
    """Fair value gaps."""

    enabled: bool = True
    max_tracked: int = Field(default=3, ge=1, le=10)
    min_size_atr: float = Field(default=0.15, ge=0.0)
    max_age_bars: int = Field(default=288, gt=0)


class OrderBlockFeaturesConfig(StrictModel):
    """Order blocks."""

    enabled: bool = True
    max_tracked: int = Field(default=2, ge=1, le=10)
    displacement_atr: float = Field(default=1.0, gt=0.0)
    max_age_bars: int = Field(default=288, gt=0)


class PatternFeaturesConfig(StrictModel):
    """Continuous candle-pattern strengths in ``[0, 1]``."""

    enabled: bool = True
    breakout_period: int = Field(default=20, gt=1)
    momentum_period: int = Field(default=5, gt=1)
    compression_period: int = Field(default=7, gt=1)


class TimeFeaturesConfig(StrictModel):
    """Cyclical time encodings and session context."""

    enabled: bool = True
    session_vol_days: int = Field(default=20, gt=1)


class MtfFeaturesConfig(StrictModel):
    """Higher-timeframe context.

    ``suffix_separator`` only affects naming. The alignment rule itself is not
    configurable: HTF features are always joined with ``merge_asof`` on
    ``close_time``, backward, exact matches allowed.
    """

    enabled: bool = True
    timeframes: tuple[Timeframe, ...] = (Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1)
    groups: tuple[str, ...] = ("price", "volatility", "trend", "structure")
    include_running_position: bool = True
    # Shorter EMAs on higher timeframes on purpose: an EMA(200) on the daily
    # frame needs 200 *days* of history, which would leave the column entirely
    # NaN on anything under a year of data and silently waste a feature slot.
    htf_ema_periods: tuple[int, ...] = (20, 50)
    htf_percentile_windows: tuple[int, ...] = (100,)


class FeaturesConfig(StrictModel):
    """Root of the feature configuration.

    The hash of this object is the ``fs_version``: changing any parameter here
    writes a new feature-set directory instead of mutating an existing one.
    """

    warmup_policy: Literal["drop", "mark"] = "drop"
    max_warmup_bars: int = Field(default=1200, gt=0)
    gap_flag_bars: int = Field(default=12, ge=0)
    price: PriceFeaturesConfig = PriceFeaturesConfig()
    volatility: VolatilityFeaturesConfig = VolatilityFeaturesConfig()
    volume: VolumeFeaturesConfig = VolumeFeaturesConfig()
    trend: TrendFeaturesConfig = TrendFeaturesConfig()
    structure: StructureFeaturesConfig = StructureFeaturesConfig()
    liquidity: LiquidityFeaturesConfig = LiquidityFeaturesConfig()
    fvg: FvgFeaturesConfig = FvgFeaturesConfig()
    order_blocks: OrderBlockFeaturesConfig = OrderBlockFeaturesConfig()
    patterns: PatternFeaturesConfig = PatternFeaturesConfig()
    time: TimeFeaturesConfig = TimeFeaturesConfig()
    mtf: MtfFeaturesConfig = MtfFeaturesConfig()

    @model_validator(mode="after")
    def _primary_atr_is_computed(self) -> FeaturesConfig:
        if self.volatility.primary_atr_period not in self.volatility.atr_periods:
            raise ValueError(
                f"volatility.primary_atr_period ({self.volatility.primary_atr_period}) must be one "
                f"of atr_periods {self.volatility.atr_periods}: it is the denominator for every "
                "ATR-normalised feature in the system"
            )
        return self


# --------------------------------------------------------------------------
# Root
# --------------------------------------------------------------------------
class AppConfig(StrictModel):
    """Root configuration object."""

    project_name: str = "xaubot"
    seed: int = Field(default=42, ge=0, lt=2**32)
    log_level: str = "INFO"
    paths: PathsConfig = PathsConfig()
    instrument: InstrumentConfig = InstrumentConfig()
    calendar: CalendarConfig = CalendarConfig()
    sessions: SessionsConfig = SessionsConfig()
    data: DataConfig
    features: FeaturesConfig = FeaturesConfig()

    @model_validator(mode="after")
    def _symbols_agree(self) -> AppConfig:
        if self.instrument.symbol != self.data.symbol:
            raise ValueError(
                f"instrument.symbol ({self.instrument.symbol}) != data.symbol ({self.data.symbol})"
            )
        return self
