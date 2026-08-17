"""Feature manifest.

The manifest answers, months later, "what exactly was this model trained on?".
It is content-addressed by the feature config, so changing any parameter yields
a new ``fs_version`` and writes to a new directory instead of quietly mutating
an existing feature set.

It also carries ``max_lookback_bars``, which the walk-forward splitter uses to
compute the purge width. That is why the number is derived from the transform
declarations rather than written down by hand: adding a feature with a longer
window silently widens the purge, instead of silently leaking across folds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xaubot.config.hashing import config_hash, version_tag
from xaubot.core.enums import Timeframe
from xaubot.core.errors import FeatureError
from xaubot.features.base import FeatureSpec


@dataclass(slots=True)
class FeatureManifest:
    """Complete description of one materialised feature set."""

    fs_version: str
    config_hash: str
    symbol: str
    base_timeframe: Timeframe
    context_timeframes: tuple[Timeframe, ...]
    specs: tuple[FeatureSpec, ...]
    max_lookback_bars: int
    warmup_bars: int
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def by_group(self) -> dict[str, list[str]]:
        """Feature names grouped by owning transform."""
        grouped: dict[str, list[str]] = {}
        for spec in self.specs:
            grouped.setdefault(spec.group, []).append(spec.name)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        return {
            "fs_version": self.fs_version,
            "config_hash": self.config_hash,
            "symbol": self.symbol,
            "base_timeframe": self.base_timeframe.value,
            "context_timeframes": [tf.value for tf in self.context_timeframes],
            "n_features": self.n_features,
            "max_lookback_bars": self.max_lookback_bars,
            "warmup_bars": self.warmup_bars,
            "created_at": self.created_at,
            "features": [spec.to_dict() for spec in self.specs],
            "groups": {group: len(names) for group, names in self.by_group().items()},
            "config": self.config,
            "notes": self.notes,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> FeatureManifest:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            fs_version=payload["fs_version"],
            config_hash=payload["config_hash"],
            symbol=payload["symbol"],
            base_timeframe=Timeframe(payload["base_timeframe"]),
            context_timeframes=tuple(Timeframe(tf) for tf in payload["context_timeframes"]),
            specs=tuple(
                FeatureSpec(
                    name=item["name"],
                    group=item["group"],
                    lookback=item["lookback"],
                    description=item["description"],
                    pit_note=item.get("pit_note", ""),
                )
                for item in payload["features"]
            ),
            max_lookback_bars=payload["max_lookback_bars"],
            warmup_bars=payload["warmup_bars"],
            config=payload.get("config", {}),
            created_at=payload.get("created_at", ""),
            notes=payload.get("notes", {}),
        )

    def assert_compatible(self, columns: list[str]) -> None:
        """Verify a feature matrix matches this manifest.

        Used at inference: a model served features in a different order, or
        missing a column, produces confident nonsense rather than an error.
        """
        expected = list(self.names)
        if columns != expected:
            missing = sorted(set(expected) - set(columns))
            extra = sorted(set(columns) - set(expected))
            raise FeatureError(
                f"Feature matrix does not match manifest {self.fs_version}: "
                f"missing={missing[:10]} unexpected={extra[:10]} "
                f"(expected {len(expected)} columns, got {len(columns)})"
            )


def build_manifest(
    specs: tuple[FeatureSpec, ...],
    features_config: Any,
    symbol: str,
    base_timeframe: Timeframe,
    context_timeframes: tuple[Timeframe, ...],
    max_lookback_bars: int,
    warmup_bars: int,
    notes: dict[str, Any] | None = None,
) -> FeatureManifest:
    """Construct a manifest and derive its content-addressed version tag."""
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise FeatureError(f"Duplicate feature names would silently overwrite each other: {duplicates}")

    payload = features_config.model_dump(mode="json") if hasattr(features_config, "model_dump") else {}
    return FeatureManifest(
        fs_version=version_tag("fs", payload),
        config_hash=config_hash(payload),
        symbol=symbol,
        base_timeframe=base_timeframe,
        context_timeframes=context_timeframes,
        specs=specs,
        max_lookback_bars=max_lookback_bars,
        warmup_bars=warmup_bars,
        config=payload,
        created_at=datetime.now(UTC).isoformat(),
        notes=notes or {},
    )
