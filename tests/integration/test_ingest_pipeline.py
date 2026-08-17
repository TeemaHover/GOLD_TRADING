"""End-to-end ingestion: CSV on disk -> validated bars -> Parquet store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import make_bars
from xaubot.config.schema import (
    AppConfig,
    CsvSourceConfig,
    DataConfig,
    PathsConfig,
    ValidationConfig,
)
from xaubot.core.enums import Timeframe
from xaubot.core.errors import DataQualityError
from xaubot.core.time_utils import CLOSE_TIME
from xaubot.data.pipeline import ingest
from xaubot.data.store import read_bars, store_summary


class TestIngest:
    def test_full_pipeline(self, app_config: AppConfig) -> None:
        result = ingest(app_config)

        assert len(result.base) > 0
        assert set(result.context) == set(app_config.data.context_timeframes)
        assert result.report.rows_out == len(result.base)
        assert result.report.source_sha256
        assert result.written

    def test_round_trips_through_the_store(self, app_config: AppConfig) -> None:
        result = ingest(app_config)
        reread = read_bars(app_config.paths.canonical(), "XAUUSD", Timeframe.M5)

        assert len(reread) == len(result.base)
        assert reread.close_times.equals(result.base.close_times)
        pd.testing.assert_series_equal(reread.df["close"], result.base.df["close"], check_names=False)

    def test_store_summary_lists_every_timeframe(self, app_config: AppConfig) -> None:
        ingest(app_config)
        canonical = store_summary(app_config.paths.canonical())
        resampled = store_summary(app_config.paths.resampled())

        assert canonical["timeframe"].tolist() == ["5m"]
        assert set(resampled["timeframe"]) == {"15m", "1h", "4h", "1d"}

    def test_no_persist_writes_nothing(self, app_config: AppConfig) -> None:
        result = ingest(app_config, persist=False)
        assert result.written == []
        assert not app_config.paths.canonical().exists()

    def test_higher_timeframes_stay_causal(self, app_config: AppConfig) -> None:
        """Cross-check of the leakage guard at the pipeline level."""
        result = ingest(app_config, persist=False)
        for timeframe, frame in result.context.items():
            assert (frame.df[CLOSE_TIME] <= result.base.end).all(), timeframe
            assert frame.df.index.is_monotonic_increasing
            assert not frame.df.index.has_duplicates

    def test_bad_data_fails_loudly(self, tmp_path: Path) -> None:
        """A file with impossible bars must stop the pipeline, not be repaired
        into something that trains fine and means nothing."""
        frame = make_bars(500)
        frame.loc[100, "close"] = -5.0
        target = tmp_path / "bad.csv"
        frame.rename(columns={"open_time": "datetime"}).drop(columns=[CLOSE_TIME]).to_csv(target, index=False)

        config = AppConfig(
            paths=PathsConfig(data_root=tmp_path / "data"),
            data=DataConfig(
                source=CsvSourceConfig(path=target, timeframe=Timeframe.M5),
                validation=ValidationConfig(fail_on_error=True),
            ),
        )
        with pytest.raises(DataQualityError, match="error-severity"):
            ingest(config, persist=False, hash_source=False)

    def test_completeness_floor_is_enforced(self, tmp_path: Path) -> None:
        frame = make_bars(2000)
        # Remove a third of the bars from the middle: a catastrophic outage.
        frame = frame.drop(index=range(500, 1200)).reset_index(drop=True)
        target = tmp_path / "gappy.csv"
        frame.rename(columns={"open_time": "datetime"}).drop(columns=[CLOSE_TIME]).to_csv(target, index=False)

        config = AppConfig(
            paths=PathsConfig(data_root=tmp_path / "data"),
            data=DataConfig(
                source=CsvSourceConfig(path=target, timeframe=Timeframe.M5),
                validation=ValidationConfig(min_completeness=0.95),
            ),
        )
        with pytest.raises(DataQualityError, match="completeness"):
            ingest(config, persist=False, hash_source=False)
