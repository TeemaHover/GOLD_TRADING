"""The feature engine.

One implementation, two drivers:

- :meth:`FeatureEngine.transform` replays the whole history (training, backtest)
- :class:`StreamingFeatureEngine` feeds one bar at a time (paper, live)

The streaming driver does **not** reimplement anything. It keeps a trailing
buffer of history and calls the same batch transforms on it, taking the final
row. Parity between backtest and live therefore is not a property that has to
be maintained by discipline -- it reduces to a single testable claim:

    transform(history)[t]  ==  transform(history.as_of(t))[-1]

which is exactly the property that fails when a transform reaches forward or
uses a statistic computed over the whole series. That test lives in
``tests/leakage/test_feature_pit.py``.

Recomputing ~250 features over a 1,500-bar buffer takes well under the 5-minute
bar interval, so there is no reason to trade correctness for an incremental
implementation that could drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from xaubot.config.schema import AppConfig, FeaturesConfig
from xaubot.core.calendar import TradingCalendar
from xaubot.core.enums import Timeframe
from xaubot.core.errors import FeatureError
from xaubot.core.logging import get_logger
from xaubot.core.sessions import SessionWindow, classify_sessions
from xaubot.core.types import BarFrame
from xaubot.features._ta import atr as atr_fn
from xaubot.features.base import FeatureContext, FeatureSpec, Transform
from xaubot.features.fvg import FvgTransform
from xaubot.features.liquidity import LiquidityTransform
from xaubot.features.manifest import FeatureManifest, build_manifest
from xaubot.features.mtf import align_htf_features, cross_timeframe_features, running_htf_features
from xaubot.features.order_blocks import OrderBlockTransform
from xaubot.features.patterns import PatternTransform
from xaubot.features.price import PriceTransform
from xaubot.features.structure import StructureTransform
from xaubot.features.time_features import TimeTransform
from xaubot.features.trend import TrendTransform
from xaubot.features.volatility import VolatilityTransform
from xaubot.features.volume import VolumeTransform

logger = get_logger(__name__)


@dataclass(slots=True)
class FeatureMatrix:
    """Computed features plus the manifest describing them."""

    values: pd.DataFrame
    manifest: FeatureManifest
    warmup_dropped: int = 0

    def __len__(self) -> int:
        return len(self.values)

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.values.index[0])

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.values.index[-1])

    def nan_report(self) -> pd.Series:
        """Fraction of NaN per column, worst first."""
        return (self.values.isna().mean().sort_values(ascending=False)).rename("nan_fraction")


class FeatureEngine:
    """Builds the full feature matrix for one symbol."""

    def __init__(
        self,
        config: FeaturesConfig,
        calendar: TradingCalendar,
        session_windows: tuple[SessionWindow, ...],
        symbol: str = "XAUUSD",
    ) -> None:
        self.config = config
        self.calendar = calendar
        self.session_windows = session_windows
        self.symbol = symbol

    # -- construction -----------------------------------------------------
    def _base_transforms(self) -> list[Transform]:
        """Transforms applied to the execution timeframe."""
        cfg = self.config
        transforms: list[Transform] = []
        if cfg.price.enabled:
            transforms.append(PriceTransform(cfg.price.return_periods, cfg.price.zscore_periods))
        if cfg.volatility.enabled:
            transforms.append(VolatilityTransform(cfg.volatility))
        if cfg.volume.enabled:
            transforms.append(VolumeTransform(cfg.volume))
        if cfg.trend.enabled:
            transforms.append(TrendTransform(cfg.trend))
        if cfg.structure.enabled:
            transforms.append(StructureTransform(cfg.structure))
        if cfg.liquidity.enabled:
            transforms.append(LiquidityTransform(cfg.liquidity))
        if cfg.fvg.enabled:
            transforms.append(FvgTransform(cfg.fvg))
        if cfg.order_blocks.enabled:
            transforms.append(OrderBlockTransform(cfg.order_blocks))
        if cfg.patterns.enabled:
            transforms.append(PatternTransform(cfg.patterns))
        if cfg.time.enabled:
            transforms.append(TimeTransform(cfg.time))
        return transforms

    def _htf_transforms(self) -> list[Transform]:
        """Reduced transform set for context timeframes.

        Deliberately smaller than the base set. Sweep detection and candle
        patterns on a 4h bar describe something a 5m entry cannot act on, and
        every extra HTF column costs warmup measured in *base* bars: an EMA(200)
        on the daily frame needs 200 days of history.
        """
        cfg = self.config
        groups = set(cfg.mtf.groups)
        transforms: list[Transform] = []

        if "price" in groups and cfg.price.enabled:
            transforms.append(PriceTransform((1, 3, 5), (20,)))
        if "volatility" in groups and cfg.volatility.enabled:
            volatility = cfg.volatility.model_copy(
                update={
                    "percentile_windows": cfg.mtf.htf_percentile_windows,
                    "bb_percentile_window": max(cfg.mtf.htf_percentile_windows),
                }
            )
            transforms.append(VolatilityTransform(volatility))
        if "trend" in groups and cfg.trend.enabled:
            trend = cfg.trend.model_copy(
                update={
                    "ema_periods": cfg.mtf.htf_ema_periods,
                    "linreg_periods": (20,),
                    "hurst_period": 50,
                }
            )
            transforms.append(TrendTransform(trend))
        if "structure" in groups and cfg.structure.enabled:
            transforms.append(StructureTransform(cfg.structure))
        return transforms

    def _context(self, bars: pd.DataFrame, timeframe: Timeframe) -> FeatureContext:
        """Build the shared context, computing the canonical ATR once."""
        period = self.config.volatility.primary_atr_period
        atr = atr_fn(bars["high"], bars["low"], bars["close"], period)
        sessions = classify_sessions(pd.DatetimeIndex(bars.index), self.session_windows)
        return FeatureContext(
            timeframe=timeframe,
            atr=atr,
            sessions=sessions,
            calendar=self.calendar,
            config=self.config,
        )

    # -- lookback accounting ----------------------------------------------
    def required_warmup_bars(self, base_timeframe: Timeframe) -> int:
        """Warmup requirement expressed in *base* bars.

        A 200-period EMA on the 4h frame is 200 four-hour bars, which is 9,600
        five-minute bars. Reporting HTF lookbacks in their own units is how a
        feature set ends up quietly producing an all-NaN column.
        """
        base = max((t.lookback for t in self._base_transforms()), default=1)

        htf = 0
        if self.config.mtf.enabled:
            htf_lookback = max((t.lookback for t in self._htf_transforms()), default=1)
            for timeframe in self.config.mtf.timeframes:
                ratio = timeframe.minutes // base_timeframe.minutes
                htf = max(htf, htf_lookback * ratio)

        return max(base, htf)

    # -- batch driver ------------------------------------------------------
    def transform(
        self,
        base: BarFrame,
        context: dict[Timeframe, BarFrame] | None = None,
        *,
        drop_warmup: bool | None = None,
        verify_alignment: bool = True,
    ) -> FeatureMatrix:
        """Compute the full feature matrix.

        Args:
            base: Execution-timeframe bars.
            context: Higher-timeframe bars, keyed by timeframe.
            drop_warmup: Override the configured warmup policy.
            verify_alignment: Assert that no HTF join read an unclosed bar.

        Returns:
            A :class:`FeatureMatrix`.
        """
        context = context or {}
        bars = base.df
        ctx = self._context(bars, base.timeframe)

        frames: list[pd.DataFrame] = []
        specs: list[FeatureSpec] = []

        for transform in self._base_transforms():
            frames.append(transform.compute(bars, ctx))
            specs.extend(transform.specs)

        if self.config.mtf.enabled and context:
            htf_frames, htf_specs = self._compute_context(context, base)
            if htf_frames:
                aligned = align_htf_features(
                    pd.DatetimeIndex(bars.index), htf_frames, verify=verify_alignment
                )
                frames.append(aligned)
                specs.extend(htf_specs)

        merged = pd.concat(frames, axis=1)

        if self.config.mtf.enabled and self.config.mtf.include_running_position:
            running = running_htf_features(bars, self.config.mtf.timeframes, ctx.atr, base.timeframe)
            merged = pd.concat([merged, running], axis=1)
            specs.extend(
                FeatureSpec(
                    name=column,
                    group="mtf_running",
                    lookback=1,
                    description="state of the higher-timeframe bar currently in progress",
                    pit_note="aggregated from completed base bars, never read off the HTF frame",
                )
                for column in running.columns
            )

        if self.config.mtf.enabled and context:
            cross = cross_timeframe_features(
                merged, self.config.mtf.timeframes, merged.get("ema_stack_score")
            )
            if not cross.empty:
                merged = pd.concat([merged, cross], axis=1)
                specs.extend(
                    FeatureSpec(
                        name=column,
                        group="mtf_cross",
                        lookback=1,
                        description="agreement or ratio across timeframes",
                    )
                    for column in cross.columns
                )

        required = self.required_warmup_bars(base.timeframe)
        warmup = min(required, self.config.max_warmup_bars)
        if required > self.config.max_warmup_bars:
            logger.warning(
                "Feature warmup needs %d base bars but max_warmup_bars is %d. Keeping the extra "
                "rows; expect NaNs in long-lookback columns, which the dataset stage will drop "
                "using TRAIN-fold statistics only.",
                required,
                self.config.max_warmup_bars,
            )

        manifest = build_manifest(
            specs=tuple(specs),
            features_config=self.config,
            symbol=self.symbol,
            base_timeframe=base.timeframe,
            context_timeframes=tuple(context),
            max_lookback_bars=required,
            warmup_bars=warmup,
            notes={"n_base_bars": len(bars), "volume_enabled": self.config.volume.enabled},
        )
        manifest.assert_compatible(list(merged.columns))

        should_drop = self.config.warmup_policy == "drop" if drop_warmup is None else drop_warmup
        dropped = 0
        if should_drop and warmup < len(merged):
            dropped = warmup
            merged = merged.iloc[warmup:]

        self._report_dead_columns(merged)

        logger.info(
            "Built %d features over %d bars (%s), fs_version=%s, warmup=%d bars dropped",
            merged.shape[1],
            len(merged),
            base.timeframe.value,
            manifest.fs_version,
            dropped,
        )
        return FeatureMatrix(values=merged, manifest=manifest, warmup_dropped=dropped)

    def _compute_context(
        self, context: dict[Timeframe, BarFrame], base: BarFrame
    ) -> tuple[dict[Timeframe, pd.DataFrame], list[FeatureSpec]]:
        """Compute the reduced feature set on each higher timeframe."""
        transforms = self._htf_transforms()
        frames: dict[Timeframe, pd.DataFrame] = {}
        specs: list[FeatureSpec] = []

        for timeframe in self.config.mtf.timeframes:
            frame = context.get(timeframe)
            if frame is None:
                logger.warning("Context timeframe %s requested but not supplied", timeframe.value)
                continue

            htf_ctx = self._context(frame.df, timeframe)
            parts = [transform.compute(frame.df, htf_ctx) for transform in transforms]
            if not parts:
                continue

            combined = pd.concat(parts, axis=1)
            frames[timeframe] = combined
            ratio = timeframe.minutes // base.timeframe.minutes
            for transform in transforms:
                specs.extend(
                    FeatureSpec(
                        name=f"{spec.name}_{timeframe.value}",
                        group=f"{spec.group}_{timeframe.value}",
                        # Restated in base bars so the purge width is correct.
                        lookback=spec.lookback * ratio,
                        description=f"{spec.description} ({timeframe.value} context)",
                        pit_note="as-of joined on close_time; only completed HTF bars are visible",
                    )
                    for spec in transform.specs
                )

        return frames, specs

    @staticmethod
    def _report_dead_columns(features: pd.DataFrame) -> None:
        """Warn about columns that carry no information.

        Not dropped here: which columns to keep must be decided on training-fold
        statistics only (see docs/ARCHITECTURE.md section 6.2), and a feature set
        whose column list depends on the data it was run over is not reproducible.
        """
        all_nan = [column for column in features.columns if features[column].isna().all()]
        if all_nan:
            logger.warning(
                "%d feature(s) are entirely NaN - almost always insufficient history for their "
                "lookback: %s%s",
                len(all_nan),
                all_nan[:8],
                "..." if len(all_nan) > 8 else "",
            )

        constant = [
            column
            for column in features.columns
            if column not in all_nan and features[column].dropna().to_numpy().std() == 0.0
        ]
        if constant:
            logger.warning(
                "%d feature(s) are constant and carry no signal: %s%s",
                len(constant),
                constant[:8],
                "..." if len(constant) > 8 else "",
            )


class StreamingFeatureEngine:
    """One-bar-at-a-time driver for paper and live trading.

    Holds a trailing buffer and delegates to the batch engine, so live features
    are computed by exactly the code the backtest used. Any incremental
    optimisation would reintroduce the possibility of drift between the two,
    which is the failure mode that makes a live system quietly stop matching
    its backtest.
    """

    def __init__(
        self,
        engine: FeatureEngine,
        base_timeframe: Timeframe,
        buffer_multiplier: float = 1.25,
    ) -> None:
        self.engine = engine
        self.base_timeframe = base_timeframe
        required = engine.required_warmup_bars(base_timeframe)
        self.buffer_size = int(max(required, engine.config.max_warmup_bars) * buffer_multiplier) + 10
        self._base: pd.DataFrame | None = None
        self._context: dict[Timeframe, BarFrame] = {}

    def prime(self, base: BarFrame, context: dict[Timeframe, BarFrame] | None = None) -> None:
        """Seed the buffer with history before the first live bar."""
        self._base = base.df.iloc[-self.buffer_size :].copy()
        self._context = dict(context or {})
        logger.info("Streaming engine primed with %d bars", len(self._base))

    def update(
        self, bar: pd.Series | pd.DataFrame, context: dict[Timeframe, BarFrame] | None = None
    ) -> pd.Series:
        """Append a closed bar and return its feature row.

        Args:
            bar: The newly closed base bar, carrying the canonical columns.
            context: Refreshed higher-timeframe bars. Only *completed* HTF bars
                may be supplied; the resampler already drops the trailing
                incomplete one.

        Returns:
            The feature row for this bar.
        """
        if self._base is None:
            raise FeatureError("StreamingFeatureEngine.update() called before prime()")

        incoming = bar.to_frame().T if isinstance(bar, pd.Series) else bar
        # Transposing a Series yields object-dtype columns, which silently
        # poisons the whole buffer on concat and breaks any numpy ufunc
        # downstream. Restore the buffer's dtypes explicitly.
        incoming = incoming.astype(self._base.dtypes.to_dict())
        self._base = pd.concat([self._base, incoming]).iloc[-self.buffer_size :]
        if context:
            self._context = dict(context)

        buffered = BarFrame(df=self._base, timeframe=self.base_timeframe, symbol=self.engine.symbol)
        matrix = self.engine.transform(buffered, self._context, drop_warmup=False, verify_alignment=True)
        return matrix.values.iloc[-1]


def build_engine(config: AppConfig, calendar: TradingCalendar) -> FeatureEngine:
    """Construct a :class:`FeatureEngine` from the application config."""
    windows = tuple(
        SessionWindow.from_hhmm(window.session, window.tz, window.start, window.end)
        for window in config.sessions.windows
        if window.enabled
    )
    return FeatureEngine(
        config=config.features,
        calendar=calendar,
        session_windows=windows,
        symbol=config.data.symbol,
    )
