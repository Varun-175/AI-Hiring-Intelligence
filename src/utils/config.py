from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


# ==========================================================
# YAML Loader
# ==========================================================

@lru_cache(maxsize=16)
def load_yaml_config(path: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Features
    --------
    ✓ Cached (disk read only once)
    ✓ Safe defaults
    ✓ UTF-8 support
    ✓ Returns {} instead of None
    """

    config_path = Path(path)

    if not config_path.exists():
        return {} if default is None else dict(default)

    try:

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = yaml.safe_load(file)

            if data is None:
                return {} if default is None else dict(default)

            if not isinstance(data, dict):
                raise ValueError(
                    f"{path} must contain a YAML mapping."
                )

            return data

    except Exception as exc:

        print(f"[Config] Failed to load '{path}': {exc}")

        return {} if default is None else dict(default)


# ==========================================================
# Nested Lookup
# ==========================================================

def get_nested(
    config: Dict[str, Any],
    keys: Iterable[str],
    default: Any = None,
) -> Any:
    """
    Retrieve nested values safely.

    Example
    -------
    get_nested(cfg, ["models", "cross_encoder"])
    """

    value = config

    for key in keys:

        if not isinstance(value, dict):
            return default

        value = value.get(key)

        if value is None:
            return default

    return value


# ==========================================================
# Config Merge
# ==========================================================

def merge_dicts(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recursively merge dictionaries.
    """

    result = dict(base)

    for key, value in override.items():

        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):

            result[key] = merge_dicts(
                result[key],
                value,
            )

        else:

            result[key] = value

    return result


# ==========================================================
# Cache Management
# ==========================================================

def clear_config_cache() -> None:
    """
    Clears cached YAML files.

    Useful for tests.
    """

    load_yaml_config.cache_clear()


# ==========================================================
# Diagnostics
# ==========================================================

def config_exists(path: str) -> bool:
    return Path(path).exists()