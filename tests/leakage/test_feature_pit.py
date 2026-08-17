"""Point-in-time guards for the feature engine.

The single most important test in this file is
:meth:`TestReplayEquivalence.test_every_feature_survives_truncated_replay`. It
asserts the property that makes the whole design work:

    transform(history)[t]  ==  transform(history.as_of(t))[-1]

If that holds for every column, then no feature can depend on data that did not
exist at decision time -- and, because the streaming driver *is* a truncated
replay, live features equal backtest features by construction rather than by
discipline.

This is not a hypothetical safeguard. It caught a real bug during development:
``prev_day_high`` matched its period stamp exactly, so the final bar of each day
saw its own day's high, and under truncation the *in-progress* day looked
complete. The batch path, having future bars, hid the defect entirely -- it
would have surfaced only as unexplained live-vs-backtest divergence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars
from xaubot.config.schema import FeaturesConfig, MtfFeaturesConfig
from xaubot.core.calendar import DEFAULT_CALENDAR
from xaubot.core.enums import Timeframe
from xaubot.core.errors import LeakageError
from xaubot.core.sessions import DEFAULT_SESSIONS
from xaubot.core.time_utils import CLOSE_TIME
from xaubot.core.types import BarFrame
from xaubot.data.resample import resample_bars
from xaubot.features.engine import FeatureEngine, StreamingFeatureEngine
from xaubot.features.pit_audit import audit_replay_equivalence, scan_source_for_lookahead

pytestmark = pytest.mark.leakage

#: Trimmed config: 15m/1h context only, so the suite stays fast while still
#: exercising every code path that touches higher-timeframe alignment.
FAST_CONFIG = FeaturesConfig(
    mtf=MtfFeaturesConfig(timeframes=(Timeframe.M15, Timeframe.H1)),
    max_warmup_bars=400,
)


@pytest.fixture(scope="module")
def bars() -> BarFrame:
    frame = make_bars(4000, start="2026-01-05 00:00").set_index(CLOSE_TIME, drop=False)
    frame.index.name = CLOSE_TIME
    return BarFrame(df=frame, timeframe=Timeframe.M5, symbol="XAUUSD")


@pytest.fixture(scope="module")
def context(bars: BarFrame) -> dict[Timeframe, BarFrame]:
    return {tf: resample_bars(bars, tf) for tf in (Timeframe.M15, Timeframe.H1)}


@pytest.fixture(scope="module")
def engine() -> FeatureEngine:
    return FeatureEngine(FAST_CONFIG, DEFAULT_CALENDAR, DEFAULT_SESSIONS)


class TestStaticScan:
    def test_features_package_has_no_forward_looking_constructs(self) -> None:
        """No ``shift(-n)``, ``center=True``, or ``bfill`` in the features package.

        Each of those is legitimate elsewhere -- label construction genuinely
        does look forward -- which is why the scan is scoped to this package.
        """
        package = Path(__file__).resolve().parents[2] / "src" / "xaubot" / "features"
        assert package.is_dir(), package

        findings = scan_source_for_lookahead(package)
        assert not findings, "forward-looking constructs found:\n" + "\n".join(str(f) for f in findings)

    def test_the_scanner_actually_catches_things(self, tmp_path: Path) -> None:
        """A guard that always passes is worse than no guard."""
        offender = tmp_path / "leaky.py"
        offender.write_text(
            "import pandas as pd\n"
            "def bad(s):\n"
            "    a = s.shift(-1)\n"
            "    b = s.rolling(5, center=True).mean()\n"
            "    c = s.bfill()\n"
            "    return a, b, c\n",
            encoding="utf-8",
        )
        findings = scan_source_for_lookahead(tmp_path)
        constructs = {f.construct for f in findings}
        assert "shift(negative)" in constructs
        assert "center=True" in constructs
        assert "bfill()" in constructs


class TestReplayEquivalence:
    def test_every_feature_survives_truncated_replay(
        self, engine: FeatureEngine, bars: BarFrame, context: dict[Timeframe, BarFrame]
    ) -> None:
        result = audit_replay_equivalence(engine, bars, context, n_cutoffs=4, seed=3, raise_on_failure=False)
        assert result.passed, result.summary() + f"\nfailing: {sorted(result.mismatches)[:20]}"
        assert result.columns_tested > 100, "audit should cover the whole feature set"

    def test_the_audit_actually_catches_a_leak(
        self, bars: BarFrame, context: dict[Timeframe, BarFrame]
    ) -> None:
        """Inject a deliberately clairvoyant feature and confirm it is caught."""

        class LeakyEngine(FeatureEngine):
            def transform(self, base, ctx=None, **kwargs):  # type: ignore[no-untyped-def]
                matrix = super().transform(base, ctx, **kwargs)
                # A statistic over the whole series: its value at any bar
                # depends on bars that had not happened yet.
                matrix.values["leaky_global_mean"] = base.df["close"].mean()
                return matrix

        leaky = LeakyEngine(FAST_CONFIG, DEFAULT_CALENDAR, DEFAULT_SESSIONS)
        with pytest.raises(LeakageError, match="leaky_global_mean"):
            audit_replay_equivalence(leaky, bars, context, n_cutoffs=2, raise_on_failure=True)


class TestHtfAlignment:
    def test_no_feature_row_reads_an_unclosed_htf_bar(
        self, engine: FeatureEngine, bars: BarFrame, context: dict[Timeframe, BarFrame]
    ) -> None:
        """``verify_alignment`` raises on violation, so a clean run is the assertion."""
        matrix = engine.transform(bars, context, drop_warmup=False, verify_alignment=True)
        assert matrix.values.shape[0] == len(bars)

    def test_htf_context_is_piecewise_constant_between_closes(
        self, engine: FeatureEngine, bars: BarFrame, context: dict[Timeframe, BarFrame]
    ) -> None:
        """A 1h feature may only change on bars where a 1h bar has just closed.

        If it changes mid-hour, something is reading the bar in progress.
        """
        matrix = engine.transform(bars, context, drop_warmup=False)
        column = next(c for c in matrix.values.columns if c.endswith("_1h") and "ema_stack" in c)

        values = matrix.values[column]
        changed = values.ne(values.shift(1)) & values.notna() & values.shift(1).notna()
        change_times = pd.DatetimeIndex(matrix.values.index[changed])

        htf_closes = set(context[Timeframe.H1].close_times)
        offenders = [t for t in change_times if t not in htf_closes]
        assert not offenders, f"{column} changed at {offenders[:5]}, which are not 1h close times"

    def test_running_features_are_not_read_off_the_htf_frame(
        self, engine: FeatureEngine, bars: BarFrame, context: dict[Timeframe, BarFrame]
    ) -> None:
        """A running 1h range must never exceed the completed bar's true range.

        Reading the HTF frame directly would give the finished bar's high and
        low from the first minute of the hour; building from completed 5m bars
        makes the range grow monotonically instead.
        """
        matrix = engine.transform(bars, context, drop_warmup=False)
        running = matrix.values["running_range_atr_1h"].dropna()
        progress = matrix.values["running_progress_1h"].reindex(running.index)

        # At the very start of an HTF bar the running range must be small;
        # if it already equalled the full bar's range, the future leaked in.
        early = running[progress <= 1 / 12]
        late = running[progress >= 0.9]
        assert early.median() < late.median()


class TestStreamingParity:
    def test_streaming_matches_batch_bar_for_bar(
        self, engine: FeatureEngine, bars: BarFrame, context: dict[Timeframe, BarFrame]
    ) -> None:
        """Live features must equal backtest features exactly.

        Divergence here is the classic reason a live system quietly stops
        behaving like its backtest.
        """
        batch = engine.transform(bars, context, drop_warmup=False)

        split = len(bars) - 5
        history = BarFrame(df=bars.df.iloc[:split], timeframe=Timeframe.M5, symbol="XAUUSD")

        streaming = StreamingFeatureEngine(engine, Timeframe.M5)
        streaming.prime(history, context)

        for position in range(split, len(bars)):
            timestamp = bars.close_times[position]
            live_context = {tf: frame.as_of(timestamp) for tf, frame in context.items()}
            row = streaming.update(bars.df.iloc[position], live_context)

            expected = batch.values.loc[timestamp]
            for column in batch.values.columns:
                left, right = expected[column], row[column]
                if pd.isna(left) and pd.isna(right):
                    continue
                assert not (pd.isna(left) ^ pd.isna(right)), f"{column} NaN mismatch at {timestamp}"
                np.testing.assert_allclose(
                    float(right), float(left), rtol=1e-6, atol=1e-8, err_msg=f"{column} @ {timestamp}"
                )


class TestLevelCausality:
    def test_previous_day_high_never_reflects_the_current_day(self, bars: BarFrame) -> None:
        """The bug the replay audit originally caught, pinned as its own test."""
        from tests.unit.test_feature_groups import context_for
        from xaubot.features.liquidity import LiquidityTransform

        frame = bars.df
        ctx = context_for(frame)
        levels = LiquidityTransform(FAST_CONFIG.liquidity)._build_levels(frame, ctx)

        trading_day = pd.Series(
            DEFAULT_CALENDAR.trading_day(pd.DatetimeIndex(frame.index)), index=frame.index
        )
        running_high_today = frame["high"].groupby(trading_day).cummax()

        valid = levels["prev_day_high"].notna()
        # The previous day's high is a fixed number for the whole day, so it
        # cannot track today's running high.
        correlation = levels.loc[valid, "prev_day_high"].groupby(trading_day[valid]).nunique().max()
        assert correlation == 1, "prev_day_high changed within a single trading day"
        assert not levels.loc[valid, "prev_day_high"].equals(running_high_today[valid])
