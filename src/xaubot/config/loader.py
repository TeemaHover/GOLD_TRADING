"""Configuration loading.

Resolution order, later wins:

1. ``config/base.yaml`` (defaults)
2. any additional YAML files passed explicitly
3. environment variables (``XAUBOT_*``) -- the only source of secrets
4. dotted-key CLI overrides (``--set data.source.path=...``)

The resolved config is validated by pydantic and then frozen, so nothing can
mutate it at runtime and produce a result that no longer matches its recorded
hash.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from xaubot.config.schema import AppConfig
from xaubot.core.errors import ConfigError
from xaubot.core.logging import get_logger

logger = get_logger(__name__)

CONFIG_DIR_ENV = "XAUBOT_CONFIG_DIR"
DEFAULT_CONFIG_DIR = Path("config")

#: Environment variables mapped onto config paths. Secrets are deliberately
#: absent -- they are consumed directly by the execution layer, never stored
#: in the config object where they could be serialised into an artifact.
ENV_MAP: dict[str, tuple[str, ...]] = {
    "XAUBOT_DATA_ROOT": ("paths", "data_root"),
    "XAUBOT_ARTIFACT_ROOT": ("paths", "artifact_root"),
    "XAUBOT_LOG_LEVEL": ("log_level",),
    "XAUBOT_SEED": ("seed",),
}


def load_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file into a dict."""
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config file {path} must contain a mapping at the top level")
    return loaded


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def set_dotted(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``a.b.c = value`` inside a nested dict, creating intermediate dicts."""
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


def _coerce_scalar(raw: str) -> Any:
    """Interpret a CLI/env string as YAML so ``true``/``3``/``[a,b]`` work."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_env(config: dict[str, Any], environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Overlay ``XAUBOT_*`` environment variables onto a config dict."""
    env = os.environ if environ is None else environ
    result = deepcopy(config)
    for var, path in ENV_MAP.items():
        if (raw := env.get(var)) not in (None, ""):
            set_dotted(result, ".".join(path), _coerce_scalar(raw))
            logger.debug("Config override from %s -> %s", var, ".".join(path))
    return result


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Overlay dotted-key overrides (values may be raw strings)."""
    if not overrides:
        return config
    result = deepcopy(config)
    for key, value in overrides.items():
        set_dotted(result, key, _coerce_scalar(value) if isinstance(value, str) else value)
    return result


def config_dir() -> Path:
    """Directory holding the YAML config files."""
    return Path(os.environ.get(CONFIG_DIR_ENV, str(DEFAULT_CONFIG_DIR)))


def load_config(
    *extra_files: Path | str,
    overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
    base: Path | str | None = None,
) -> AppConfig:
    """Load, merge, and validate the full application config.

    Args:
        *extra_files: Additional YAML files layered over the base config.
        overrides: Dotted-key overrides applied last.
        environ: Environment mapping (defaults to :data:`os.environ`).
        base: Path to the base YAML file. Defaults to ``<config_dir>/base.yaml``.

    Raises:
        ConfigError: If a file is missing or validation fails.
    """
    base_path = Path(base) if base is not None else config_dir() / "base.yaml"
    merged = load_yaml(base_path)

    for extra in extra_files:
        merged = deep_merge(merged, load_yaml(Path(extra)))

    merged = apply_env(merged, environ)
    merged = apply_overrides(merged, overrides)

    try:
        config = AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration:\n{exc}") from exc

    logger.debug("Loaded config from %s (+%d overlay files)", base_path, len(extra_files))
    return config
