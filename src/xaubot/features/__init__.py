"""Feature engineering.

One implementation drives both batch (training, backtest) and streaming
(paper, live) feature computation; see :mod:`xaubot.features.engine`.

This package may not import from ``models``, ``signals``, ``risk``,
``backtesting``, or ``execution``. Dependencies flow strictly downward.
"""

from __future__ import annotations

from xaubot.features.base import FeatureContext, FeatureSpec, Transform
from xaubot.features.engine import (
    FeatureEngine,
    FeatureMatrix,
    StreamingFeatureEngine,
    build_engine,
)
from xaubot.features.manifest import FeatureManifest, build_manifest
from xaubot.features.pit_audit import (
    audit_replay_equivalence,
    scan_source_for_lookahead,
)

__all__ = [
    "FeatureContext",
    "FeatureEngine",
    "FeatureManifest",
    "FeatureMatrix",
    "FeatureSpec",
    "StreamingFeatureEngine",
    "Transform",
    "audit_replay_equivalence",
    "build_engine",
    "build_manifest",
    "scan_source_for_lookahead",
]
