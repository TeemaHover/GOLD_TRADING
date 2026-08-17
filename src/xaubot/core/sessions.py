"""Trading-session classification.

Sessions are defined in their own *local* exchange timezone rather than as
fixed UTC windows. This matters: London and New York both observe DST, and they
switch on different dates, so a hard-coded UTC window is wrong for several weeks
of every year -- and those weeks are exactly when the London/NY overlap (the
highest-volatility period for gold) shifts by an hour.

All classification is a pure function of the bar's own close time, so it is
inherently point-in-time safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xaubot.core.enums import Session


@dataclass(frozen=True, slots=True)
class SessionWindow:
    """A session defined by local wall-clock hours in a named timezone.

    Attributes:
        session: The session label produced for bars inside this window.
        tz: IANA timezone name, e.g. ``"Europe/London"``.
        start_minute: Minutes past local midnight when the session opens.
        end_minute: Minutes past local midnight when it closes (exclusive).
            May exceed 1440 to express a window that wraps past midnight.
        weekdays: Local weekdays on which the session runs (0=Monday).
    """

    session: Session
    tz: str
    start_minute: int
    end_minute: int
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)

    @classmethod
    def from_hhmm(
        cls,
        session: Session,
        tz: str,
        start: str,
        end: str,
        weekdays: tuple[int, ...] = (0, 1, 2, 3, 4),
    ) -> SessionWindow:
        """Build from ``"HH:MM"`` strings, e.g. ``from_hhmm(..., "08:00", "16:30")``."""
        start_m = _hhmm_to_minutes(start)
        end_m = _hhmm_to_minutes(end)
        if end_m <= start_m:
            end_m += 1440  # wraps past local midnight
        return cls(session=session, tz=tz, start_minute=start_m, end_minute=end_m, weekdays=weekdays)

    def contains(self, utc_index: pd.DatetimeIndex) -> np.ndarray:
        """Vectorised membership test for a UTC index."""
        local = utc_index.tz_convert(self.tz)
        minutes = local.hour * 60 + local.minute
        weekday = local.weekday

        # Same-day portion of the window.
        inside = (minutes >= self.start_minute) & (minutes < min(self.end_minute, 1440))
        inside &= np.isin(weekday, self.weekdays)

        if self.end_minute > 1440:
            # Wrapped portion belongs to the *previous* local day's session.
            wrapped = minutes < (self.end_minute - 1440)
            prev_weekday = (weekday - 1) % 7
            wrapped &= np.isin(prev_weekday, self.weekdays)
            inside |= wrapped

        return np.asarray(inside)

    def session_date(self, utc_index: pd.DatetimeIndex) -> pd.Index:
        """Local calendar date each timestamp's session belongs to.

        Used to group bars into discrete session instances (for session
        high/low, session range, "previous session" levels, and so on).
        """
        local = utc_index.tz_convert(self.tz)
        minutes = local.hour * 60 + local.minute
        # Timestamps inside a wrapped tail belong to the previous local day.
        offset = np.where(
            (self.end_minute > 1440) & (minutes < (self.end_minute - 1440)),
            -1,
            0,
        )
        return pd.Index(local.normalize() + pd.to_timedelta(offset, unit="D"))


def _hhmm_to_minutes(value: str) -> int:
    hours, _, minutes = value.partition(":")
    return int(hours) * 60 + int(minutes or 0)


#: Default session definitions. Asia is anchored to Tokyo (no DST), London and
#: New York to their own zones so the overlap tracks DST correctly.
DEFAULT_SESSIONS: tuple[SessionWindow, ...] = (
    SessionWindow.from_hhmm(Session.ASIA, "Asia/Tokyo", "09:00", "18:00"),
    SessionWindow.from_hhmm(Session.LONDON, "Europe/London", "08:00", "16:30"),
    SessionWindow.from_hhmm(Session.NY, "America/New_York", "08:00", "17:00"),
)


def classify_sessions(
    utc_index: pd.DatetimeIndex,
    windows: tuple[SessionWindow, ...] = DEFAULT_SESSIONS,
) -> pd.DataFrame:
    """Label each timestamp with its session and per-session membership flags.

    Args:
        utc_index: Bar close times, tz-aware UTC.
        windows: Session definitions. Defaults to Asia/London/NY.

    Returns:
        DataFrame indexed like ``utc_index`` with columns:

        - ``session``: the primary :class:`~xaubot.core.enums.Session` label.
          When London and New York are both active the label is
          ``LONDON_NY_OVERLAP``; when nothing is active it is ``OFF``.
        - ``in_asia`` / ``in_london`` / ``in_ny``: boolean membership.
        - ``session_id``: stable identifier of the *session instance*, used for
          grouping (e.g. ``"LONDON:2026-08-12"``).
    """
    membership: dict[Session, np.ndarray] = {w.session: w.contains(utc_index) for w in windows}
    by_session = {w.session: w for w in windows}

    in_asia = membership.get(Session.ASIA, np.zeros(len(utc_index), dtype=bool))
    in_london = membership.get(Session.LONDON, np.zeros(len(utc_index), dtype=bool))
    in_ny = membership.get(Session.NY, np.zeros(len(utc_index), dtype=bool))

    label = np.full(len(utc_index), Session.OFF.value, dtype=object)
    # Precedence: overlap beats NY beats London beats Asia.
    label[in_asia] = Session.ASIA.value
    label[in_london] = Session.LONDON.value
    label[in_ny] = Session.NY.value
    label[in_london & in_ny] = Session.LONDON_NY_OVERLAP.value

    session_id = np.full(len(utc_index), "OFF", dtype=object)
    for session in (Session.ASIA, Session.LONDON, Session.NY):
        window = by_session.get(session)
        if window is None:
            continue
        mask = membership[session]
        if not mask.any():
            continue
        dates = window.session_date(utc_index).strftime("%Y-%m-%d")
        session_id[mask] = np.char.add(f"{session.value}:", np.asarray(dates, dtype=str))[mask]
    # Overlap bars are grouped with the NY session instance for level tracking.

    return pd.DataFrame(
        {
            "session": pd.Categorical(label, categories=[s.value for s in Session]),
            "in_asia": in_asia,
            "in_london": in_london,
            "in_ny": in_ny,
            "session_id": session_id,
        },
        index=utc_index,
    )
