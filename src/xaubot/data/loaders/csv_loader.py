"""CSV bar loader.

Handles the formats that actually show up in practice:

- ``datetime,open,high,low,close,volume`` with ISO timestamps (tz-aware or not)
- MT5 exports with separate ``<DATE>`` and ``<TIME>`` columns, tab-separated
- ``tickvol`` / ``tick_volume`` / ``vol`` naming for the volume column
- comma decimal separators

What it will *not* do is guess whether a timestamp is bar-open or bar-close.
That comes from config, because guessing wrong shifts every feature by one bar
and produces a silent, untestable look-ahead bias.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from xaubot.config.schema import CsvSourceConfig
from xaubot.core.errors import SchemaError
from xaubot.core.logging import get_logger
from xaubot.core.time_utils import CLOSE_TIME, OPEN_TIME, derive_bar_times, ensure_utc
from xaubot.data.loaders.base import BarLoader

logger = get_logger(__name__)

#: Source column name (lower-cased, stripped of ``<>``) -> canonical name.
_CANONICAL_ALIASES: dict[str, str] = {
    "open": "open",
    "o": "open",
    "high": "high",
    "h": "high",
    "low": "low",
    "l": "low",
    "close": "close",
    "c": "close",
    "price": "close",
    "adj close": "close",
    "volume": "volume",
    "vol": "volume",
    "tickvol": "volume",
    "tick_volume": "volume",
    "tickvolume": "volume",
    "real_volume": "real_volume",
    "spread": "spread",
}

_TIMESTAMP_CANDIDATES = ("datetime", "date_time", "timestamp", "time", "date", "gmt time")
_DATE_CANDIDATES = ("date", "<date>")
_TIME_CANDIDATES = ("time", "<time>")


def _normalise_name(name: str) -> str:
    return name.strip().strip("<>").strip().lower().replace(" ", "_")


def _sniff_delimiter(path: Path, skiprows: int) -> str:
    """Detect the delimiter from a sample of the file."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for _ in range(skiprows):
            handle.readline()
        sample = handle.read(8192)
    if not sample:
        raise SchemaError(f"{path} is empty")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        # Sniffer fails on single-column-looking samples; fall back to counting.
        counts = {d: sample.count(d) for d in (",", ";", "\t", "|")}
        best = max(counts, key=lambda d: counts[d])
        return best if counts[best] else ","


class CsvBarLoader(BarLoader):
    """Load bars from a delimited text file."""

    def __init__(self, config: CsvSourceConfig, symbol: str) -> None:
        """
        Args:
            config: Source description (path, timestamp semantics, overrides).
            symbol: Instrument symbol these bars belong to.
        """
        self.config = config
        self.symbol = symbol
        self.timeframe = config.timeframe

    def describe(self) -> str:
        return f"csv:{self.config.path.as_posix()}"

    def load(self) -> pd.DataFrame:
        """Parse the file into a canonical, UTC, unrepaired bar frame."""
        path = self.config.path
        if not path.is_file():
            raise SchemaError(f"CSV source not found: {path}")

        delimiter = self.config.delimiter or _sniff_delimiter(path, self.config.skiprows)
        logger.info("Reading %s (delimiter=%r)", path, delimiter)

        frame = pd.read_csv(
            path,
            sep=delimiter,
            skiprows=self.config.skiprows or None,
            decimal=self.config.decimal,
            encoding="utf-8-sig",
            engine="python" if len(delimiter) > 1 else "c",
        )
        if frame.empty:
            raise SchemaError(f"{path} contains no data rows")

        frame = self._rename_columns(frame)
        stamps = self._build_timestamps(frame, path)
        frame = self._select_ohlcv(frame, path)

        open_time, close_time = derive_bar_times(stamps, self.timeframe, self.config.timestamp_convention)
        frame.insert(0, CLOSE_TIME, close_time)
        frame.insert(0, OPEN_TIME, open_time)

        logger.info(
            "Loaded %d rows from %s spanning %s -> %s (timestamps interpreted as bar %s)",
            len(frame),
            path.name,
            open_time.min(),
            close_time.max(),
            self.config.timestamp_convention.value,
        )
        return frame.reset_index(drop=True)

    # -- internals --------------------------------------------------------
    def _rename_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Map source column names onto canonical ones."""
        explicit = {_normalise_name(k): v for k, v in self.config.column_map.items()}
        renamed: dict[str, str] = {}
        for column in frame.columns:
            key = _normalise_name(str(column))
            if key in explicit:
                renamed[column] = explicit[key]
            elif key in _CANONICAL_ALIASES:
                renamed[column] = _CANONICAL_ALIASES[key]
            else:
                renamed[column] = key
        out = frame.rename(columns=renamed)
        if self.config.volume_column is not None:
            source = _normalise_name(self.config.volume_column)
            if source in out.columns:
                out = out.rename(columns={source: "volume"})
        return out.loc[:, ~out.columns.duplicated(keep="first")]

    def _build_timestamps(self, frame: pd.DataFrame, path: Path) -> pd.DatetimeIndex:
        """Assemble a UTC DatetimeIndex from one or two source columns."""
        cfg = self.config
        columns = {c.lower(): c for c in frame.columns.astype(str)}

        if cfg.date_column and cfg.time_column:
            date_col = columns.get(_normalise_name(cfg.date_column))
            time_col = columns.get(_normalise_name(cfg.time_column))
            if date_col is None or time_col is None:
                raise SchemaError(
                    f"{path}: date/time columns {cfg.date_column!r}/{cfg.time_column!r} not found; "
                    f"available: {sorted(frame.columns)}"
                )
            raw = frame[date_col].astype(str).str.strip() + " " + frame[time_col].astype(str).str.strip()
        else:
            ts_col = columns.get(_normalise_name(cfg.timestamp_column))
            if ts_col is None:
                for candidate in _TIMESTAMP_CANDIDATES:
                    if candidate in columns:
                        ts_col = columns[candidate]
                        logger.warning(
                            "Configured timestamp column %r not found; using %r",
                            cfg.timestamp_column,
                            ts_col,
                        )
                        break
            if ts_col is None:
                date_col = next((columns[c] for c in _DATE_CANDIDATES if c in columns), None)
                time_col = next((columns[c] for c in _TIME_CANDIDATES if c in columns), None)
                if date_col and time_col:
                    raw = (
                        frame[date_col].astype(str).str.strip()
                        + " "
                        + frame[time_col].astype(str).str.strip()
                    )
                else:
                    raise SchemaError(
                        f"{path}: no timestamp column found; available: {sorted(frame.columns)}"
                    )
            else:
                raw = frame[ts_col]

        as_utc = cfg.source_tz is None
        try:
            # Fast path: pandas infers one consistent format for the whole column.
            parsed = pd.to_datetime(raw, utc=as_utc, errors="raise")
        except (ValueError, TypeError):
            # Slow path: per-element inference, for files with ragged formats.
            logger.warning("Falling back to mixed-format timestamp parsing (slower)")
            parsed = pd.to_datetime(raw, utc=as_utc, format="mixed", errors="coerce")

        if parsed.isna().any():
            bad = int(parsed.isna().sum())
            examples = raw[parsed.isna()].astype(str).head(3).tolist()
            raise SchemaError(
                f"{path}: {bad} timestamps could not be parsed (e.g. {examples}). "
                "Set data.source.timestamp_column / delimiter explicitly."
            )
        return ensure_utc(pd.DatetimeIndex(parsed), cfg.source_tz)

    def _select_ohlcv(self, frame: pd.DataFrame, path: Path) -> pd.DataFrame:
        """Keep OHLCV as floats, synthesising volume if the feed lacks it."""
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise SchemaError(f"{path}: missing price columns {missing}; available: {sorted(frame.columns)}")

        out = frame.loc[:, required].apply(pd.to_numeric, errors="coerce").astype("float64")

        if "volume" in frame.columns:
            out["volume"] = pd.to_numeric(frame["volume"], errors="coerce").astype("float64")
        else:
            logger.warning(
                "%s has no volume column; filling with NaN. Volume features must be disabled "
                "(features.volume.enabled=false) or they will be silently useless.",
                path.name,
            )
            out["volume"] = pd.NA
            out["volume"] = out["volume"].astype("float64")

        return out
