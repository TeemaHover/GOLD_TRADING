"""Feature-set persistence.

Feature sets are written under their ``fs_version`` -- the content hash of the
feature config -- and never mutated in place. Changing any feature parameter
produces a new directory, so a model artifact can always be paired with the
exact feature definitions it was trained on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from xaubot.core.errors import StorageError
from xaubot.core.logging import get_logger
from xaubot.core.time_utils import CLOSE_TIME, to_utc_timestamp
from xaubot.features.engine import FeatureMatrix
from xaubot.features.manifest import FeatureManifest

logger = get_logger(__name__)

_COMPRESSION = "zstd"
MANIFEST_NAME = "manifest.json"


def feature_dir(root: Path, fs_version: str, symbol: str) -> Path:
    """Directory holding one materialised feature set."""
    return root / f"fs_version={fs_version}" / f"symbol={symbol}"


def write_features(matrix: FeatureMatrix, root: Path, symbol: str) -> Path:
    """Persist a feature matrix and its manifest.

    Args:
        matrix: The computed features.
        root: Feature-store root.
        symbol: Instrument symbol.

    Returns:
        The directory written to.
    """
    target = feature_dir(root, matrix.manifest.fs_version, symbol)
    target.mkdir(parents=True, exist_ok=True)

    frame = matrix.values.copy()
    frame.insert(0, CLOSE_TIME, pd.DatetimeIndex(frame.index))
    frame = frame.reset_index(drop=True)

    stamps = pd.DatetimeIndex(frame[CLOSE_TIME])
    written = 0
    for (year, month), chunk in frame.groupby([stamps.year, stamps.month], sort=True):
        partition = target / f"year={year}" / f"month={month:02d}"
        partition.mkdir(parents=True, exist_ok=True)
        destination = partition / "part.parquet"
        tmp = destination.with_suffix(".parquet.tmp")
        try:
            pq.write_table(pa.Table.from_pandas(chunk, preserve_index=False), tmp, compression=_COMPRESSION)
            tmp.replace(destination)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise StorageError(f"Failed writing features to {destination}: {exc}") from exc
        written += 1

    matrix.manifest.save(target / MANIFEST_NAME)
    logger.info(
        "Wrote %d features x %d bars to %s (%d partitions)",
        matrix.values.shape[1],
        len(matrix.values),
        target,
        written,
    )
    return target


def read_features(
    root: Path,
    fs_version: str,
    symbol: str,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, FeatureManifest]:
    """Read a feature set back, verifying it matches its manifest.

    Raises:
        StorageError: If the feature set or its manifest is missing.
    """
    source = feature_dir(root, fs_version, symbol)
    manifest_path = source / MANIFEST_NAME
    if not manifest_path.is_file():
        raise StorageError(f"No feature manifest at {manifest_path}")

    manifest = FeatureManifest.load(manifest_path)

    # Read the parquet parts explicitly rather than scanning the directory:
    # the manifest lives alongside them, and a dataset scan would try to parse
    # manifest.json as parquet.
    parts = sorted(source.rglob("*.parquet"))
    if not parts:
        raise StorageError(f"No feature partitions under {source}")

    try:
        table = pa.concat_tables([pq.read_table(part) for part in parts])
        frame = table.to_pandas()
    except Exception as exc:
        raise StorageError(f"Failed reading features from {source}: {exc}") from exc

    frame[CLOSE_TIME] = pd.to_datetime(frame[CLOSE_TIME], utc=True)
    frame = frame.sort_values(CLOSE_TIME).set_index(CLOSE_TIME)
    frame.index.name = CLOSE_TIME

    if start is not None:
        frame = frame.loc[to_utc_timestamp(start) :]
    if end is not None:
        frame = frame.loc[: to_utc_timestamp(end)]

    manifest.assert_compatible(list(frame.columns))
    return frame, manifest


def list_feature_sets(root: Path) -> pd.DataFrame:
    """Summarise every feature set present in the store."""
    rows: list[dict[str, object]] = []
    if not root.exists():
        return pd.DataFrame(columns=["fs_version", "symbol", "n_features", "rows", "created_at"])

    for version_dir in sorted(root.glob("fs_version=*")):
        fs_version = version_dir.name.split("=", 1)[1]
        for symbol_dir in sorted(version_dir.glob("symbol=*")):
            manifest_path = symbol_dir / MANIFEST_NAME
            if not manifest_path.is_file():
                continue
            manifest = FeatureManifest.load(manifest_path)
            total = sum(pq.read_metadata(f).num_rows for f in symbol_dir.rglob("*.parquet"))
            rows.append(
                {
                    "fs_version": fs_version,
                    "symbol": symbol_dir.name.split("=", 1)[1],
                    "n_features": manifest.n_features,
                    "rows": total,
                    "created_at": manifest.created_at,
                }
            )
    return pd.DataFrame(rows)
