"""Exception hierarchy.

Every failure mode the pipeline can hit has a named exception, so that callers
can distinguish "your CSV is malformed" from "your config is wrong" from
"the code has a bug". Bare ``Exception`` is never raised.
"""

from __future__ import annotations


class XaubotError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(XaubotError):
    """Configuration is missing, malformed, or internally inconsistent."""


class DataError(XaubotError):
    """Base class for data-pipeline failures."""


class SchemaError(DataError):
    """Input data does not match the expected column schema."""


class DataQualityError(DataError):
    """Data quality is below the configured acceptance threshold."""


class ResampleError(DataError):
    """A timeframe conversion is invalid (e.g. downsampling to a finer grid)."""


class StorageError(DataError):
    """A read from or write to the Parquet/relational store failed."""


class LeakageError(XaubotError):
    """A point-in-time violation was detected.

    This is deliberately fatal. Look-ahead bias silently invalidates every
    downstream result, so the pipeline stops rather than producing numbers that
    look good and mean nothing.
    """


class FeatureError(XaubotError):
    """A feature transform failed or produced an invalid result."""


class ModelError(XaubotError):
    """Model construction, training, or inference failed."""


class RiskViolationError(XaubotError):
    """An action was blocked by the risk engine."""
