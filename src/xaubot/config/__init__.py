"""Typed configuration: schema, loading, content-addressed hashing."""

from __future__ import annotations

from xaubot.config.hashing import config_hash, file_sha256, version_tag
from xaubot.config.loader import load_config, load_yaml
from xaubot.config.schema import AppConfig, DataConfig, InstrumentConfig, PathsConfig

__all__ = [
    "AppConfig",
    "DataConfig",
    "InstrumentConfig",
    "PathsConfig",
    "config_hash",
    "file_sha256",
    "load_config",
    "load_yaml",
    "version_tag",
]
