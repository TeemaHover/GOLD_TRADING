"""Ingestion orchestrator.

Wires the stages together in the only order that is correct:

``load -> validate (report) -> clean (repair) -> BarFrame (enforce invariants)
-> gap analysis -> resample -> persist``

Validation runs *before* cleaning so the report describes the source as it
arrived. Gap analysis runs *after* cleaning so it measures real feed outages
rather than rows that were dropped for being malformed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from xaubot.config.hashing import file_sha256
from xaubot.config.schema import AppConfig
from xaubot.core.calendar import TradingCalendar
from xaubot.core.enums import Timeframe
from xaubot.core.errors import DataQualityError
from xaubot.core.logging import get_logger
from xaubot.core.types import BarFrame, QualityReport
from xaubot.data.cleaning import clean_bars
from xaubot.data.loaders.csv_loader import CsvBarLoader
from xaubot.data.resample import resample_many
from xaubot.data.store import write_bars, write_json
from xaubot.data.validators import detect_gaps, detect_issues

logger = get_logger(__name__)


@dataclass(slots=True)
class IngestResult:
    """Everything one ingestion run produced."""

    base: BarFrame
    context: dict[Timeframe, BarFrame]
    report: QualityReport
    written: list[Path]

    def summary(self) -> str:
        lines = [
            f"{self.report.symbol} {self.report.timeframe.value}: "
            f"{self.report.rows_out:,} bars  {self.report.start} -> {self.report.end}",
            f"  dropped {self.report.rows_dropped:,} rows, {len(self.report.issues)} issue types",
        ]
        if self.report.gaps is not None:
            gaps = self.report.gaps
            lines.append(
                f"  completeness {gaps.completeness:.2%} "
                f"({gaps.missing_bars:,} missing of {gaps.expected_bars:,} expected, "
                f"largest gap {gaps.largest_gap_bars} bars)"
            )
        for tf, frame in self.context.items():
            lines.append(f"  {tf.value}: {len(frame):,} bars")
        return "\n".join(lines)


def build_calendar(config: AppConfig) -> TradingCalendar:
    """Construct the trading calendar from config."""
    cal = config.calendar
    holidays = frozenset(pd.Timestamp(d).date() for d in cal.holidays)
    return TradingCalendar(
        tz=cal.tz,
        week_open_weekday=cal.week_open_weekday,
        week_open_time=cal.week_open_time,
        week_close_weekday=cal.week_close_weekday,
        week_close_time=cal.week_close_time,
        daily_break_start=cal.daily_break_start,
        daily_break_end=cal.daily_break_end,
        holidays=holidays,
    )


def ingest(
    config: AppConfig,
    *,
    persist: bool = True,
    hash_source: bool = True,
) -> IngestResult:
    """Run the full ingestion pipeline.

    Args:
        config: Resolved application config.
        persist: Write canonical and resampled bars to the Parquet store.
        hash_source: Compute the source file's SHA-256 for provenance. Skipped
            in tests where the extra read is not worth the time.

    Returns:
        An :class:`IngestResult`.

    Raises:
        DataQualityError: If the data fails the configured acceptance
            thresholds and ``validation.fail_on_error`` is set. Continuing with
            known-bad data produces results that look fine and mean nothing.
    """
    data_cfg = config.data
    calendar = build_calendar(config)

    loader = CsvBarLoader(data_cfg.source, symbol=data_cfg.symbol)
    raw = loader.load()
    rows_in = len(raw)

    issues = detect_issues(raw, data_cfg.base_timeframe, data_cfg.validation)
    for issue in issues:
        logger.log(
            {"INFO": 20, "WARNING": 30, "ERROR": 40}[issue.severity.value],
            "[%s] %s x%d - %s",
            issue.severity.value,
            issue.kind.value,
            issue.count,
            issue.detail,
        )

    cleaned = clean_bars(raw, data_cfg.base_timeframe, data_cfg.cleaning)
    base = BarFrame(df=cleaned.frame, timeframe=data_cfg.base_timeframe, symbol=data_cfg.symbol)

    gaps = detect_gaps(
        base.close_times,
        data_cfg.base_timeframe,
        calendar,
        max_gaps_reported=data_cfg.validation.max_gap_bars_reported,
    )

    report = QualityReport(
        symbol=data_cfg.symbol,
        timeframe=data_cfg.base_timeframe,
        source=loader.describe(),
        rows_in=rows_in,
        rows_out=len(base),
        start=base.start,
        end=base.end,
        issues=tuple(issues),
        gaps=gaps,
        source_sha256=file_sha256(data_cfg.source.path) if hash_source else "",
    )

    _enforce_quality(report, config)

    context = resample_many(base, data_cfg.context_timeframes, data_cfg.resample, calendar)

    written: list[Path] = []
    if persist:
        canonical_root = config.paths.canonical()
        written.extend(write_bars(base, canonical_root))
        resampled_root = config.paths.resampled()
        for frame in context.values():
            written.extend(write_bars(frame, resampled_root))
        write_json(
            report.to_dict(),
            config.paths.reports() / f"quality_{data_cfg.symbol}_{data_cfg.base_timeframe.value}.json",
        )

    return IngestResult(base=base, context=context, report=report, written=written)


def _enforce_quality(report: QualityReport, config: AppConfig) -> None:
    """Fail loudly when the data is not fit for training."""
    validation = config.data.validation
    problems: list[str] = []

    if report.has_errors:
        kinds = [i.kind.value for i in report.issues if i.severity.value == "ERROR"]
        problems.append(f"error-severity issues remain: {kinds}")

    if report.gaps is not None and report.gaps.completeness < validation.min_completeness:
        problems.append(
            f"completeness {report.gaps.completeness:.2%} is below the required "
            f"{validation.min_completeness:.2%}"
        )

    if not problems:
        return

    message = "Data quality check failed:\n  - " + "\n  - ".join(problems)
    if validation.fail_on_error:
        raise DataQualityError(
            message + "\n\nFix the source data, or relax data.validation.* if the shortfall is understood. "
            "Training on data you know is broken wastes the run."
        )
    logger.warning("%s (continuing because fail_on_error=false)", message)
