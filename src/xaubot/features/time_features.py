"""Time, session, and calendar-position features.

Time is encoded cyclically. Feeding ``hour`` as 0-23 tells a model that 23:00
and 00:00 are 23 units apart, when they are adjacent; ``(sin, cos)`` pairs make
that adjacency geometric, which matters here because gold's character genuinely
changes at session boundaries rather than at midnight.

``bars_since_gap`` earns its place from the data itself: 16 of the 86
extreme-return bars in the current dataset are the first bar after the daily
maintenance break, where an overnight move is compressed into one 5m candle
(docs/DATA_CONTRACT.md). Without this flag, the model would learn that "the
01:00 bar is explosive" as if it were a tradable property rather than an
artefact of the market being closed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaubot.core.enums import Session
from xaubot.features.base import EPS, FeatureContext, FeatureSpec, Transform

_SESSIONS = (
    Session.ASIA,
    Session.LONDON,
    Session.NY,
    Session.LONDON_NY_OVERLAP,
    Session.OFF,
)


class TimeTransform(Transform):
    """Cyclical clock encodings, session identity, and session-relative context."""

    group = "time"

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        self.cfg = config

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        cfg = self.cfg
        out: list[FeatureSpec] = [
            FeatureSpec("hour_sin", self.group, 1, "cyclical hour encoding"),
            FeatureSpec("hour_cos", self.group, 1, "cyclical hour encoding"),
            FeatureSpec("minute_of_day_sin", self.group, 1, "cyclical minute-of-day encoding"),
            FeatureSpec("minute_of_day_cos", self.group, 1, "cyclical minute-of-day encoding"),
            FeatureSpec("dow_sin", self.group, 1, "cyclical day-of-week encoding"),
            FeatureSpec("dow_cos", self.group, 1, "cyclical day-of-week encoding"),
            FeatureSpec("month_sin", self.group, 1, "cyclical month encoding"),
            FeatureSpec("month_cos", self.group, 1, "cyclical month encoding"),
            FeatureSpec("day_of_week", self.group, 1, "0=Monday .. 6=Sunday"),
            FeatureSpec("is_month_end", self.group, 1, "final trading day of the month"),
            FeatureSpec("is_quarter_end", self.group, 1, "final trading day of the quarter"),
        ]
        out += [
            FeatureSpec(f"session_is_{s.value.lower()}", self.group, 1, f"current session is {s.value}")
            for s in _SESSIONS
        ]
        out += [
            FeatureSpec("session_progress", self.group, 1, "fraction of the current session elapsed, [0,1]"),
            FeatureSpec("bars_since_session_open", self.group, 1, "bars since the current session began"),
            FeatureSpec("is_session_first_hour", self.group, 1, "within the first hour of the session"),
            FeatureSpec("is_session_last_hour", self.group, 1, "within the final hour of the session"),
            FeatureSpec(
                "session_range_atr", self.group, 1, "running range of the current session, in ATR units"
            ),
            # Distances to the running session high/low are emitted by the
            # liquidity transform, which owns every reference level. Duplicating
            # them here would give the model two identical columns.
            FeatureSpec(
                "session_atr_ratio",
                self.group,
                288 * cfg.session_vol_days,
                "current session ATR vs this session's trailing average",
                pit_note="trailing average excludes the session in progress",
            ),
            FeatureSpec(
                "session_trend", self.group, 1, "normalised drift within the current session, [-1,1]"
            ),
            FeatureSpec(
                "bars_since_gap",
                self.group,
                1,
                "bars since the last break in the feed (weekend, maintenance, outage)",
                pit_note="bars just after a gap span closed-market time; their returns are not 5m returns",
            ),
            FeatureSpec(
                "is_post_gap",
                self.group,
                1,
                "within the configured window after a feed gap - exclude from labels",
            ),
        ]
        return tuple(out)

    def _compute(self, bars: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        cfg = self.cfg
        index = pd.DatetimeIndex(bars.index)
        sessions = ctx.sessions.reindex(bars.index)
        atr = ctx.atr
        out = pd.DataFrame(index=bars.index)

        minute_of_day = index.hour * 60 + index.minute
        out["hour_sin"] = np.sin(2 * np.pi * index.hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * index.hour / 24.0)
        out["minute_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
        out["minute_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
        out["dow_sin"] = np.sin(2 * np.pi * index.dayofweek / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * index.dayofweek / 7.0)
        out["month_sin"] = np.sin(2 * np.pi * index.month / 12.0)
        out["month_cos"] = np.cos(2 * np.pi * index.month / 12.0)
        out["day_of_week"] = index.dayofweek.astype("float64")

        # Month/quarter end by calendar position rather than "is this the last
        # bar of the month". The latter is only knowable once the month has
        # ended, which makes it unavailable at decision time by construction.
        out["is_month_end"] = (index.day >= 28).astype("float64")
        out["is_quarter_end"] = ((index.month % 3 == 0) & (index.day >= 28)).astype("float64")

        session = sessions["session"].astype(str)
        for value in _SESSIONS:
            out[f"session_is_{value.value.lower()}"] = (session == value.value).astype("float64")

        session_id = sessions["session_id"]
        position = bars.groupby(session_id).cumcount()
        size = session_id.map(session_id.value_counts())
        out["bars_since_session_open"] = position.astype("float64")
        # Progress uses the *typical* length of this session type rather than
        # this instance's final length, which is not knowable mid-session.
        typical_bars = float(np.median(size[size > 1])) if (size > 1).any() else 1.0
        out["session_progress"] = (position / max(typical_bars, 1.0)).clip(0.0, 1.0)

        bars_per_hour = max(1, 60 // ctx.timeframe.minutes)
        out["is_session_first_hour"] = (position < bars_per_hour).astype("float64")
        out["is_session_last_hour"] = (out["session_progress"] > 0.85).astype("float64")

        running_high = bars["high"].groupby(session_id).cummax()
        running_low = bars["low"].groupby(session_id).cummin()
        session_open = bars["open"].groupby(session_id).transform("first")
        out["session_range_atr"] = (running_high - running_low) / (atr + EPS)
        out["session_trend"] = np.tanh((bars["close"] - session_open) / (atr + EPS) / 3.0)

        out["session_atr_ratio"] = self._session_atr_ratio(atr, session, cfg.session_vol_days)

        gap_bars = self._bars_since_gap(index, ctx)
        out["bars_since_gap"] = gap_bars
        out["is_post_gap"] = (gap_bars < ctx.config.gap_flag_bars).astype("float64")

        return out

    @staticmethod
    def _session_atr_ratio(atr: pd.Series, session: pd.Series, days: int) -> pd.Series:
        """Current ATR against this session's own trailing average.

        Compared within session type, because a London ATR is not comparable to
        an Asian one. The baseline is shifted so the session in progress does
        not normalise itself.
        """
        window = max(2, days * 288 // max(len(session.unique()), 1))
        baseline = atr.groupby(session, group_keys=False).apply(
            lambda group: group.shift(1).rolling(window, min_periods=max(10, window // 10)).mean()
        )
        return atr / (baseline.reindex(atr.index) + EPS)

    @staticmethod
    def _bars_since_gap(index: pd.DatetimeIndex, ctx: FeatureContext) -> pd.Series:
        """Bars elapsed since the last discontinuity in the feed.

        A gap is any step larger than one bar duration. Weekend and maintenance
        breaks count: what matters downstream is that the bar's return spans
        closed-market time, not why the market was closed.
        """
        expected = pd.Timedelta(ctx.timeframe.duration)
        deltas = pd.Series(index, index=index).diff()
        is_gap = np.array(deltas > expected, dtype=bool)  # copy: pandas views are read-only
        is_gap[0] = True  # the first bar has no predecessor

        counter = np.zeros(len(index))
        since = 0
        for i in range(len(index)):
            since = 0 if is_gap[i] else since + 1
            counter[i] = since
        return pd.Series(counter, index=index)
