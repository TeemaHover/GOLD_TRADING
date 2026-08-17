"""End-to-end: bars -> features -> Parquet -> back, with the manifest enforced."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import make_bars
from xaubot.config.schema import AppConfig, FeaturesConfig, MtfFeaturesConfig
from xaubot.core.calendar import DEFAULT_CALENDAR
from xaubot.core.enums import Timeframe
from xaubot.core.errors import FeatureError
from xaubot.core.sessions import DEFAULT_SESSIONS
from xaubot.core.time_utils import CLOSE_TIME
from xaubot.core.types import BarFrame
from xaubot.data.resample import resample_bars
from xaubot.features.engine import FeatureEngine, build_engine
from xaubot.features.manifest import FeatureManifest
from xaubot.features.store import list_feature_sets, read_features, write_features

CONFIG = FeaturesConfig(
    mtf=MtfFeaturesConfig(timeframes=(Timeframe.M15, Timeframe.H1)),
    max_warmup_bars=300,
)


@pytest.fixture(scope="module")
def bars() -> BarFrame:
    frame = make_bars(2500, start="2026-01-05 00:00").set_index(CLOSE_TIME, drop=False)
    frame.index.name = CLOSE_TIME
    return BarFrame(df=frame, timeframe=Timeframe.M5, symbol="XAUUSD")


@pytest.fixture(scope="module")
def context(bars: BarFrame) -> dict[Timeframe, BarFrame]:
    return {tf: resample_bars(bars, tf) for tf in (Timeframe.M15, Timeframe.H1)}


@pytest.fixture(scope="module")
def engine() -> FeatureEngine:
    return FeatureEngine(CONFIG, DEFAULT_CALENDAR, DEFAULT_SESSIONS)


class TestFeatureMatrix:
    def test_shape_and_alignment(
        self, engine: FeatureEngine, bars: BarFrame, context: dict[Timeframe, BarFrame]
    ) -> None:
        matrix = engine.transform(bars, context)
        assert matrix.values.shape[1] == matrix.manifest.n_features
        assert matrix.values.index.is_monotonic_increasing
        assert not matrix.values.index.has_duplicates
        assert matrix.values.index.isin(bars.close_times).all()

    def test_warmup_rows_are_dropped(
        self, engine: FeatureEngine, bars: BarFrame, context: dict[Timeframe, BarFrame]
    ) -> None:
        matrix = engine.transform(bars, context)
        assert matrix.warmup_dropped > 0
        assert len(matrix) == len(bars) - matrix.warmup_dropped

    def test_all_columns_are_float64(
        self, engine: FeatureEngine, bars: BarFrame, context: dict[Timeframe, BarFrame]
    ) -> None:
        """Mixed dtypes reach the model as silently-cast garbage."""
        matrix = engine.transform(bars, context)
        assert (matrix.values.dtypes == "float64").all()

    def test_no_infinities(
        self, engine: FeatureEngine, bars: BarFrame, context: dict[Timeframe, BarFrame]
    ) -> None:
        """An inf survives scaling and turns a whole training batch into NaN."""
        import numpy as np

        matrix = engine.transform(bars, context)
        infinite = np.isinf(matrix.values.to_numpy(dtype="float64", na_value=0.0))
        offenders = matrix.values.columns[infinite.any(axis=0)].tolist()
        assert not offenders, f"infinite values in {offenders[:10]}"

    def test_manifest_version_is_deterministic(self, bars: BarFrame, context) -> None:  # type: ignore[no-untyped-def]
        """Same config -> same fs_version, so results stay reproducible."""
        first = FeatureEngine(CONFIG, DEFAULT_CALENDAR, DEFAULT_SESSIONS).transform(bars, context)
        second = FeatureEngine(CONFIG, DEFAULT_CALENDAR, DEFAULT_SESSIONS).transform(bars, context)
        assert first.manifest.fs_version == second.manifest.fs_version

    def test_changing_config_changes_the_version(self, bars: BarFrame, context) -> None:  # type: ignore[no-untyped-def]
        """Otherwise a changed feature set would silently overwrite an old one."""
        altered = CONFIG.model_copy(
            update={"volatility": CONFIG.volatility.model_copy(update={"atr_periods": (14, 60)})}
        )
        original = FeatureEngine(CONFIG, DEFAULT_CALENDAR, DEFAULT_SESSIONS).transform(bars, context)
        changed = FeatureEngine(altered, DEFAULT_CALENDAR, DEFAULT_SESSIONS).transform(bars, context)
        assert original.manifest.fs_version != changed.manifest.fs_version


class TestFeatureStore:
    def test_round_trip(
        self,
        tmp_path: Path,
        engine: FeatureEngine,
        bars: BarFrame,
        context: dict[Timeframe, BarFrame],
    ) -> None:
        matrix = engine.transform(bars, context)
        write_features(matrix, tmp_path, "XAUUSD")

        loaded, manifest = read_features(tmp_path, matrix.manifest.fs_version, "XAUUSD")

        assert manifest.fs_version == matrix.manifest.fs_version
        assert list(loaded.columns) == list(matrix.values.columns)
        assert loaded.index.equals(matrix.values.index)
        pd.testing.assert_frame_equal(loaded, matrix.values, check_freq=False)

    def test_date_filtering(
        self,
        tmp_path: Path,
        engine: FeatureEngine,
        bars: BarFrame,
        context: dict[Timeframe, BarFrame],
    ) -> None:
        matrix = engine.transform(bars, context)
        write_features(matrix, tmp_path, "XAUUSD")

        cutoff = matrix.values.index[len(matrix) // 2]
        loaded, _ = read_features(tmp_path, matrix.manifest.fs_version, "XAUUSD", start=cutoff)
        assert loaded.index.min() >= cutoff

    def test_listing(
        self,
        tmp_path: Path,
        engine: FeatureEngine,
        bars: BarFrame,
        context: dict[Timeframe, BarFrame],
    ) -> None:
        matrix = engine.transform(bars, context)
        write_features(matrix, tmp_path, "XAUUSD")

        listed = list_feature_sets(tmp_path)
        assert len(listed) == 1
        assert listed.iloc[0]["fs_version"] == matrix.manifest.fs_version
        assert listed.iloc[0]["n_features"] == matrix.manifest.n_features

    def test_manifest_rejects_a_mismatched_matrix(
        self, tmp_path: Path, engine: FeatureEngine, bars: BarFrame, context
    ) -> None:  # type: ignore[no-untyped-def]
        """At inference, a reordered or short feature matrix must not be
        silently accepted -- the model would produce confident nonsense."""
        matrix = engine.transform(bars, context)
        manifest_path = tmp_path / "manifest.json"
        matrix.manifest.save(manifest_path)
        manifest = FeatureManifest.load(manifest_path)

        with pytest.raises(FeatureError, match="does not match manifest"):
            manifest.assert_compatible(list(matrix.values.columns)[:-1])

        reordered = list(matrix.values.columns)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with pytest.raises(FeatureError, match="does not match manifest"):
            manifest.assert_compatible(reordered)


class TestEngineFromConfig:
    def test_build_engine_uses_configured_sessions(self, tmp_path: Path) -> None:
        from xaubot.config.schema import CsvSourceConfig, DataConfig, PathsConfig

        config = AppConfig(
            paths=PathsConfig(data_root=tmp_path),
            data=DataConfig(source=CsvSourceConfig(path=tmp_path / "unused.csv")),
            features=CONFIG,
        )
        engine = build_engine(config, DEFAULT_CALENDAR)
        assert engine.symbol == "XAUUSD"
        assert len(engine.session_windows) == len(config.sessions.windows)
