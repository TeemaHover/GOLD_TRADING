"""CSV loading, validation, and cleaning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import make_bars
from xaubot.config.schema import CleaningConfig, CsvSourceConfig, ValidationConfig
from xaubot.core.calendar import DEFAULT_CALENDAR
from xaubot.core.enums import DataIssue, Severity, Timeframe, TimestampConvention
from xaubot.core.errors import SchemaError
from xaubot.core.time_utils import CLOSE_TIME, OPEN_TIME
from xaubot.data.cleaning import clean_bars
from xaubot.data.loaders.csv_loader import CsvBarLoader
from xaubot.data.validators import detect_gaps, detect_issues


def _load(path: Path, **kwargs) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    config = CsvSourceConfig(path=path, timeframe=Timeframe.M5, **kwargs)
    return CsvBarLoader(config, symbol="XAUUSD").load()


class TestCsvLoader:
    def test_loads_iso_utc_file(self, csv_path: Path) -> None:
        frame = _load(csv_path)
        assert list(frame.columns)[:2] == [OPEN_TIME, CLOSE_TIME]
        assert str(frame[CLOSE_TIME].dt.tz) == "UTC"
        assert (frame[CLOSE_TIME] - frame[OPEN_TIME] == pd.Timedelta(minutes=5)).all()

    def test_close_convention_shifts_the_bar(self, csv_path: Path) -> None:
        as_open = _load(csv_path, timestamp_convention=TimestampConvention.OPEN)
        as_close = _load(csv_path, timestamp_convention=TimestampConvention.CLOSE)
        delta = as_open[CLOSE_TIME].iloc[0] - as_close[CLOSE_TIME].iloc[0]
        assert delta == pd.Timedelta(minutes=5)

    def test_mt5_style_tab_separated_date_time(self, tmp_path: Path) -> None:
        target = tmp_path / "mt5.csv"
        target.write_text(
            "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\n"
            "2026.01.05\t10:00:00\t4000.0\t4002.0\t3999.0\t4001.0\t120\n"
            "2026.01.05\t10:05:00\t4001.0\t4003.0\t4000.0\t4002.5\t143\n",
            encoding="utf-8",
        )
        frame = _load(target, date_column="DATE", time_column="TIME", source_tz="UTC")
        assert len(frame) == 2
        assert frame["volume"].tolist() == [120.0, 143.0]
        assert frame[OPEN_TIME].iloc[0] == pd.Timestamp("2026-01-05 10:00", tz="UTC")

    def test_semicolon_and_comma_decimal(self, tmp_path: Path) -> None:
        target = tmp_path / "eu.csv"
        target.write_text(
            "datetime;open;high;low;close;volume\n"
            "2026-01-05 10:00:00;4000,5;4002,0;3999,0;4001,0;120\n"
            "2026-01-05 10:05:00;4001,0;4003,0;4000,0;4002,5;143\n",
            encoding="utf-8",
        )
        frame = _load(target, decimal=",", source_tz="UTC")
        assert frame["open"].iloc[0] == pytest.approx(4000.5)

    def test_local_timezone_conversion(self, tmp_path: Path) -> None:
        target = tmp_path / "local.csv"
        target.write_text(
            "datetime,open,high,low,close,volume\n"
            "2026-01-05 10:00:00,4000,4002,3999,4001,120\n"
            "2026-01-05 10:05:00,4001,4003,4000,4002,143\n",
            encoding="utf-8",
        )
        frame = _load(target, source_tz="Europe/London")
        assert frame[OPEN_TIME].iloc[0] == pd.Timestamp("2026-01-05 10:00", tz="UTC")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SchemaError, match="not found"):
            _load(tmp_path / "nope.csv")

    def test_missing_price_column_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "bad.csv"
        target.write_text("datetime,open,high,volume\n2026-01-05 10:00:00,1,2,3\n", encoding="utf-8")
        with pytest.raises(SchemaError, match="missing price columns"):
            _load(target)


class TestValidators:
    config = ValidationConfig()

    def test_clean_data_has_no_issues(self) -> None:
        issues = detect_issues(make_bars(500), Timeframe.M5, self.config)
        kinds = {i.kind for i in issues}
        assert DataIssue.OHLC_INCONSISTENT not in kinds
        assert DataIssue.DUPLICATE_TIMESTAMP not in kinds

    def test_detects_duplicates_and_disorder(self) -> None:
        frame = make_bars(100)
        frame = pd.concat([frame, frame.iloc[[10]]], ignore_index=True)
        issues = {i.kind: i for i in detect_issues(frame, Timeframe.M5, self.config)}
        assert issues[DataIssue.DUPLICATE_TIMESTAMP].count == 1
        assert DataIssue.NON_MONOTONIC in issues

    def test_detects_ohlc_inconsistency_as_error(self) -> None:
        frame = make_bars(100)
        frame.loc[5, "high"] = frame.loc[5, "low"] - 1.0
        issues = {i.kind: i for i in detect_issues(frame, Timeframe.M5, self.config)}
        assert issues[DataIssue.OHLC_INCONSISTENT].severity is Severity.ERROR

    def test_detects_extreme_return(self) -> None:
        frame = make_bars(500)
        frame.loc[250, ["open", "high", "low", "close"]] *= 1.5
        kinds = {i.kind for i in detect_issues(frame, Timeframe.M5, self.config)}
        assert DataIssue.EXTREME_RETURN in kinds

    def test_gap_report_ignores_the_weekend(self) -> None:
        """Synthetic bars already skip closed hours, so completeness is 100%."""
        frame = make_bars(1200)
        report = detect_gaps(pd.DatetimeIndex(frame[CLOSE_TIME]), Timeframe.M5, DEFAULT_CALENDAR)
        assert report.completeness == pytest.approx(1.0)
        assert report.missing_bars == 0

    def test_gap_report_finds_a_real_outage(self) -> None:
        frame = make_bars(1200)
        # Remove a contiguous 20-bar run from the middle of a trading day.
        frame = frame.drop(index=range(300, 320)).reset_index(drop=True)
        report = detect_gaps(pd.DatetimeIndex(frame[CLOSE_TIME]), Timeframe.M5, DEFAULT_CALENDAR)
        assert report.missing_bars == 20
        assert report.largest_gap_bars == 20
        assert report.completeness < 1.0


class TestCleaning:
    config = CleaningConfig()

    def test_sorts_and_deduplicates(self) -> None:
        frame = make_bars(200)
        shuffled = pd.concat([frame.iloc[100:], frame.iloc[:100], frame.iloc[[5]]], ignore_index=True)
        result = clean_bars(shuffled, Timeframe.M5, self.config)
        assert result.frame.index.is_monotonic_increasing
        assert not result.frame.index.has_duplicates
        assert len(result.frame) == 200

    def test_keep_last_wins_for_duplicates(self) -> None:
        frame = make_bars(50)
        dupe = frame.iloc[[10]].copy()
        # Change a field that does not break OHLC consistency, so this test
        # exercises the duplicate policy and nothing else.
        dupe.loc[:, "volume"] = 9999.0
        result = clean_bars(pd.concat([frame, dupe], ignore_index=True), Timeframe.M5, self.config)
        assert result.frame["volume"].iloc[10] == pytest.approx(9999.0)

    def test_drops_malformed_rows(self) -> None:
        frame = make_bars(100)
        frame.loc[5, "high"] = frame.loc[5, "low"] - 1.0
        frame.loc[6, "close"] = -1.0
        frame.loc[7, "open"] = pd.NA
        result = clean_bars(frame, Timeframe.M5, self.config)
        assert len(result.frame) == 97
        assert result.total_dropped == 3

    def test_does_not_fabricate_bars_by_default(self) -> None:
        frame = make_bars(200).drop(index=range(50, 55)).reset_index(drop=True)
        result = clean_bars(frame, Timeframe.M5, self.config)
        assert result.filled_bars == 0
        assert len(result.frame) == 195

    def test_opt_in_gap_filling_marks_synthetic_bars(self) -> None:
        frame = make_bars(200).drop(index=range(50, 53)).reset_index(drop=True)
        config = CleaningConfig(fill_missing_bars=True, max_forward_fill_bars=5)
        result = clean_bars(frame, Timeframe.M5, config)
        assert result.filled_bars == 3
        synthetic = result.frame.loc[result.frame["is_synthetic"]]
        assert len(synthetic) == 3
        # Synthetic bars are inert: zero range, zero volume.
        assert (synthetic["high"] == synthetic["low"]).all()
        assert (synthetic["volume"] == 0).all()

    def test_gap_filling_skips_runs_that_are_too_long(self) -> None:
        frame = make_bars(200).drop(index=range(50, 70)).reset_index(drop=True)
        config = CleaningConfig(fill_missing_bars=True, max_forward_fill_bars=5)
        result = clean_bars(frame, Timeframe.M5, config)
        assert result.filled_bars == 0
