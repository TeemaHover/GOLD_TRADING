"""Parquet bar store.

Bars live in a partitioned Parquet lake rather than a database: the access
pattern is "read a contiguous date range of a few hundred thousand rows and
hand it to numpy", which columnar files serve far better than row storage, with
no server to run.

Layout::

    <root>/symbol=XAUUSD/tf=5m/year=2026/month=03/part.parquet

Writes are per-partition and atomic (temp file + replace), so an interrupted
ingest cannot leave a half-written month behind.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from xaubot.core.enums import Timeframe
from xaubot.core.errors import StorageError
from xaubot.core.logging import get_logger
from xaubot.core.time_utils import CLOSE_TIME, OPEN_TIME, to_utc_timestamp
from xaubot.core.types import BarFrame

logger = get_logger(__name__)

_COMPRESSION = "zstd"


def partition_dir(root: Path, symbol: str, timeframe: Timeframe) -> Path:
    """Directory holding one symbol/timeframe's partitions."""
    return root / f"symbol={symbol}" / f"tf={timeframe.value}"


def write_bars(
    bars: BarFrame,
    root: Path,
    *,
    overwrite: bool = True,
) -> list[Path]:
    """Write a :class:`~xaubot.core.types.BarFrame` to the partitioned store.

    Args:
        bars: Validated bars to persist.
        root: Store root directory.
        overwrite: Replace existing partitions. When False, existing partitions
            are merged with the new rows (new rows win on duplicate timestamps).

    Returns:
        The partition file paths that were written.
    """
    base = partition_dir(root, bars.symbol, bars.timeframe)
    frame = bars.df.reset_index(drop=True)

    stamps = pd.DatetimeIndex(frame[CLOSE_TIME])
    if bars.timeframe is Timeframe.D1:
        keys = list(zip(stamps.year, [1] * len(stamps), strict=True))  # one partition per year
    else:
        keys = list(zip(stamps.year, stamps.month, strict=True))

    frame = frame.assign(_year=[k[0] for k in keys], _month=[k[1] for k in keys])
    written: list[Path] = []

    for (year, month), chunk in frame.groupby(["_year", "_month"], sort=True):
        target_dir = base / f"year={year}" / f"month={month:02d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "part.parquet"

        payload = chunk.drop(columns=["_year", "_month"])
        if target.exists() and not overwrite:
            existing = pq.read_table(target).to_pandas()
            payload = (
                pd.concat([existing, payload])
                .drop_duplicates(subset=CLOSE_TIME, keep="last")
                .sort_values(CLOSE_TIME)
            )

        _atomic_write(payload, target)
        written.append(target)

    logger.info(
        "Wrote %d %s bars to %s across %d partitions",
        len(bars),
        bars.timeframe.value,
        base,
        len(written),
    )
    return written


def _atomic_write(frame: pd.DataFrame, target: Path) -> None:
    """Write via a temp file so a crash cannot corrupt an existing partition."""
    tmp = target.with_suffix(".parquet.tmp")
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, tmp, compression=_COMPRESSION)
        tmp.replace(target)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise StorageError(f"Failed writing {target}: {exc}") from exc


def read_bars(
    root: Path,
    symbol: str,
    timeframe: Timeframe,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> BarFrame:
    """Read bars back out of the store.

    Args:
        root: Store root.
        symbol: Instrument symbol.
        timeframe: Timeframe to read.
        start: Inclusive lower bound on close time.
        end: Inclusive upper bound on close time.

    Raises:
        StorageError: If nothing has been written for this symbol/timeframe.
    """
    base = partition_dir(root, symbol, timeframe)
    if not base.exists():
        raise StorageError(f"No data in store for {symbol} {timeframe.value} (looked in {base})")

    try:
        table = pq.read_table(base)
    except Exception as exc:
        raise StorageError(f"Failed reading {base}: {exc}") from exc

    frame = table.to_pandas()
    if frame.empty:
        raise StorageError(f"Store partition {base} is empty")

    for column in (OPEN_TIME, CLOSE_TIME):
        frame[column] = pd.to_datetime(frame[column], utc=True)

    frame = frame.sort_values(CLOSE_TIME).set_index(CLOSE_TIME, drop=False)
    frame.index.name = CLOSE_TIME

    if start is not None:
        frame = frame.loc[to_utc_timestamp(start) :]
    if end is not None:
        frame = frame.loc[: to_utc_timestamp(end)]

    return BarFrame(df=frame, timeframe=timeframe, symbol=symbol)


def store_summary(root: Path) -> pd.DataFrame:
    """One row per stored symbol/timeframe: row count and date range."""
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame(columns=["symbol", "timeframe", "rows", "start", "end"])

    for symbol_dir in sorted(root.glob("symbol=*")):
        symbol = symbol_dir.name.split("=", 1)[1]
        for tf_dir in sorted(symbol_dir.glob("tf=*")):
            tf_value = tf_dir.name.split("=", 1)[1]
            files = list(tf_dir.rglob("*.parquet"))
            if not files:
                continue
            stamps: list[pd.Timestamp] = []
            total = 0
            for file in files:
                meta = pq.read_metadata(file)
                total += meta.num_rows
            column = pq.read_table(tf_dir, columns=[CLOSE_TIME]).to_pandas()[CLOSE_TIME]
            stamps = [pd.Timestamp(column.min()), pd.Timestamp(column.max())]
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": tf_value,
                    "rows": total,
                    "start": stamps[0],
                    "end": stamps[1],
                }
            )
    return pd.DataFrame(rows)


def write_json(payload: dict[str, Any], target: Path) -> None:
    """Persist a report/manifest as pretty JSON."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    logger.debug("Wrote %s", target)
