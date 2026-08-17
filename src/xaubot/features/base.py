"""Feature transform interface.

Every feature in the system is produced by a :class:`Transform` that declares,
up front and without being run, exactly which columns it emits and how many
bars of history each needs. Two things depend on that declaration:

- the **manifest** (what the model was trained on, reproducible months later)
- the **purge width** for walk-forward splitting, which is
  ``max_feature_lookback + max_label_horizon``. Hard-coding it would silently
  leak label information across fold boundaries the moment someone adds a
  longer-lookback feature.

Transforms must be *causal*: the value at bar ``i`` may depend only on bars
``<= i``. This is enforced three ways -- by review, by the AST scan in
:mod:`xaubot.features.pit_audit`, and by the replay-equivalence property test.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd

from xaubot.core.calendar import TradingCalendar
from xaubot.core.enums import Timeframe
from xaubot.core.errors import FeatureError

#: Guard against divide-by-zero in ratio features. Small enough not to distort
#: real values, large enough that float64 division stays finite.
EPS = 1e-12


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Declaration of one output column.

    Attributes:
        name: Column name. Must be unique across the whole feature set.
        group: Owning transform group (``price``, ``volatility``, ...).
        lookback: Bars of history required before the value is trustworthy.
            Rows before this are marked warmup and dropped.
        description: What the number means, carried into the manifest.
        pit_note: How this feature avoids look-ahead, when non-obvious. Present
            on anything involving confirmation delay or forward-looking shapes.
    """

    name: str
    group: str
    lookback: int
    description: str
    pit_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "lookback": self.lookback,
            "description": self.description,
            "pit_note": self.pit_note,
        }


@dataclass(slots=True)
class FeatureContext:
    """Shared state passed to every transform.

    ATR lives here rather than being recomputed per transform because nearly
    every feature normalises by it -- price distances, stop sizing, pattern
    strengths. One definition, computed once, so "1.5 ATR" means the same thing
    everywhere in the system.
    """

    timeframe: Timeframe
    atr: pd.Series
    sessions: pd.DataFrame
    calendar: TradingCalendar
    config: Any  # FeaturesConfig; typed loosely to avoid a circular import


class Transform(ABC):
    """Base class for a group of related features."""

    group: ClassVar[str] = "unknown"

    @property
    @abstractmethod
    def specs(self) -> tuple[FeatureSpec, ...]:
        """Columns this transform emits. Derived from config, never hard-coded."""

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def lookback(self) -> int:
        """Longest history requirement across this transform's outputs."""
        return max((spec.lookback for spec in self.specs), default=1)

    @abstractmethod
    def _compute(self, bars: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        """Produce the feature columns. Implemented by subclasses."""

    def compute(self, bars: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        """Compute features and verify the output matches the declaration.

        The check is not ceremony: a transform whose real output drifts from
        its declared specs produces a manifest that lies about what the model
        was trained on, and a purge width computed from stale lookbacks.
        """
        out = self._compute(bars, ctx)

        declared = list(self.names)
        missing = sorted(set(declared) - set(out.columns))
        extra = sorted(set(out.columns) - set(declared))
        if missing or extra:
            raise FeatureError(
                f"{type(self).__name__} output does not match its declared specs. "
                f"missing={missing} unexpected={extra}"
            )
        if not out.index.equals(bars.index):
            raise FeatureError(f"{type(self).__name__} changed the index; transforms must be aligned")

        # Reorder to the declared order rather than requiring transforms to
        # compute in that order. The manifest fixes column order for the model;
        # how a transform arrives at the columns is its own business.
        return out.loc[:, declared].astype("float64")
