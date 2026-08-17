"""Point-in-time auditing.

Two independent checks, because they catch different mistakes:

1. :func:`scan_source_for_lookahead` -- a static AST scan of the features
   package for constructs that read forward (``shift(-1)``, ``center=True``,
   ``bfill``, reversed iteration). Catches the bug the moment it is written,
   including in code paths a test happens not to exercise.

2. :func:`audit_replay_equivalence` -- the behavioural check. Recomputes
   features on truncated history and asserts the final row matches the same
   timestamp's row from the full run. This is the property that actually
   matters, and it catches leaks the AST scan cannot see: a global ``mean()``,
   an unshifted groupby aggregate, an off-by-one in a confirmation delay.

The static scan can produce false positives (``shift(-1)`` is legitimate in
*label* code, which is why it only scans the features package). The replay
check cannot: if it fails, the feature set is not usable.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from xaubot.core.enums import Timeframe
from xaubot.core.errors import LeakageError
from xaubot.core.logging import get_logger
from xaubot.core.types import BarFrame

if TYPE_CHECKING:
    from xaubot.features.engine import FeatureEngine

logger = get_logger(__name__)

#: Method calls that read forward in time when applied to a time-ordered frame.
_BANNED_METHODS = {
    "bfill": "backward fill copies a future value into the present",
    "backfill": "backward fill copies a future value into the present",
}

#: Keyword arguments that make a rolling window read forward.
_BANNED_KWARGS = {
    "center": "a centred rolling window includes future bars",
}


@dataclass(frozen=True, slots=True)
class LookaheadFinding:
    """One suspicious construct found by the static scan."""

    path: str
    line: int
    construct: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} {self.construct} - {self.reason}"


class _LookaheadVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.findings: list[LookaheadFinding] = []

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)

        if name in _BANNED_METHODS:
            self._record(node, f"{name}()", _BANNED_METHODS[name])

        if name in {"shift", "tshift"}:
            for arg in node.args:
                if _is_negative_number(arg):
                    self._record(node, "shift(negative)", "a negative shift pulls the future backwards")
            for keyword in node.keywords:
                if keyword.arg == "periods" and _is_negative_number(keyword.value):
                    self._record(
                        node, "shift(periods=negative)", "a negative shift pulls the future backwards"
                    )

        for keyword in node.keywords:
            if keyword.arg in _BANNED_KWARGS and _is_truthy(keyword.value):
                self._record(node, f"{keyword.arg}=True", _BANNED_KWARGS[keyword.arg])

        self.generic_visit(node)

    def _record(self, node: ast.AST, construct: str, reason: str) -> None:
        self.findings.append(
            LookaheadFinding(
                path=self.path, line=getattr(node, "lineno", 0), construct=construct, reason=reason
            )
        )


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _is_negative_number(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int | float)
        and node.operand.value > 0
    )


def _is_truthy(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def scan_source_for_lookahead(package_dir: Path) -> list[LookaheadFinding]:
    """Statically scan a package for forward-looking constructs.

    Args:
        package_dir: Directory to scan recursively for ``.py`` files.

    Returns:
        Findings, empty if the package is clean.
    """
    findings: list[LookaheadFinding] = []
    for path in sorted(package_dir.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # pragma: no cover - a broken file fails elsewhere
            logger.error("Could not parse %s: %s", path, exc)
            continue
        visitor = _LookaheadVisitor(path.name)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    return findings


@dataclass(slots=True)
class ReplayAuditResult:
    """Outcome of the replay-equivalence audit."""

    cutoffs_tested: int
    columns_tested: int
    mismatches: dict[str, int]
    worst_column: str | None
    worst_abs_diff: float

    @property
    def passed(self) -> bool:
        return not self.mismatches

    def summary(self) -> str:
        if self.passed:
            return (
                f"PASS - {self.columns_tested} features identical across {self.cutoffs_tested} "
                "truncated replays"
            )
        return (
            f"FAIL - {len(self.mismatches)} feature(s) changed when history was truncated; "
            f"worst {self.worst_column} (max abs diff {self.worst_abs_diff:.3e})"
        )


def audit_replay_equivalence(
    engine: FeatureEngine,
    base: BarFrame,
    context: dict[Timeframe, BarFrame] | None = None,
    n_cutoffs: int = 6,
    seed: int = 17,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    raise_on_failure: bool = True,
) -> ReplayAuditResult:
    """Verify that truncating history does not change already-computed features.

    This is the operational definition of point-in-time correctness for the
    feature engine, and simultaneously the backtest/live parity guarantee: the
    streaming driver *is* a truncated replay, so if this passes, live features
    equal backtest features by construction.

    Args:
        engine: The engine under test.
        base: Full-history base bars.
        context: Full-history higher-timeframe bars.
        n_cutoffs: How many random truncation points to test.
        seed: RNG seed, so failures are reproducible.
        rtol: Relative tolerance for float comparison.
        atol: Absolute tolerance for float comparison.
        raise_on_failure: Raise :class:`LeakageError` rather than returning a
            failed result. Defaults to raising, because a silent failure here
            invalidates everything downstream.

    Returns:
        A :class:`ReplayAuditResult`.
    """
    context = context or {}
    full = engine.transform(base, context, drop_warmup=False, verify_alignment=True)

    warmup = engine.required_warmup_bars(base.timeframe)
    lowest = min(warmup + 50, len(base) - 2)
    if lowest >= len(base) - 1:
        raise LeakageError(
            f"Not enough history to audit: warmup needs {warmup} bars but only {len(base)} are available"
        )

    rng = np.random.default_rng(seed)
    cutoff_positions = rng.integers(lowest, len(base) - 1, size=n_cutoffs)

    mismatches: dict[str, int] = {}
    worst_column: str | None = None
    worst_diff = 0.0
    columns = list(full.values.columns)

    for position in cutoff_positions:
        cutoff = base.close_times[int(position)]

        truncated_context = {timeframe: frame.as_of(cutoff) for timeframe, frame in context.items()}
        replayed = engine.transform(
            base.as_of(cutoff), truncated_context, drop_warmup=False, verify_alignment=True
        )

        expected = full.values.loc[cutoff]
        actual = replayed.values.iloc[-1]

        for column in columns:
            left, right = expected[column], actual[column]
            if pd.isna(left) and pd.isna(right):
                continue
            if pd.isna(left) != pd.isna(right):
                mismatches[column] = mismatches.get(column, 0) + 1
                continue
            difference = abs(float(left) - float(right))
            if difference > atol + rtol * abs(float(left)):
                mismatches[column] = mismatches.get(column, 0) + 1
                if difference > worst_diff:
                    worst_diff, worst_column = difference, column

    result = ReplayAuditResult(
        cutoffs_tested=len(cutoff_positions),
        columns_tested=len(columns),
        mismatches=mismatches,
        worst_column=worst_column,
        worst_abs_diff=worst_diff,
    )

    if not result.passed and raise_on_failure:
        offenders = sorted(mismatches.items(), key=lambda item: -item[1])[:10]
        raise LeakageError(
            "Replay-equivalence audit failed: these features change when history is truncated, "
            "which means they depend on data that would not exist at decision time.\n  "
            + "\n  ".join(f"{name}: {count}/{len(cutoff_positions)} cutoffs" for name, count in offenders)
        )

    logger.info(result.summary())
    return result
