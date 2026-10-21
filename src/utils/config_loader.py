from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def load_json_config(path: str | Path, alias_map: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Load a JSON config of the form {"parameters": [{"name": ..., "value": ...}, ...]}."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = json.loads(config_path.read_text())
    parameters = raw.get("parameters")
    if not isinstance(parameters, list):
        raise ValueError(f"Config file {config_path} must contain a 'parameters' list")

    alias_map = dict(alias_map or {})
    normalized: Dict[str, Any] = {}

    for item in parameters:
        if not isinstance(item, dict):
            raise ValueError(
                f"Each parameter entry in {config_path} must be an object with 'name' and 'value'"
            )
        if "name" not in item or "value" not in item:
            raise ValueError(f"Invalid parameter entry in {config_path}: {item!r}")

        name = str(item["name"]).strip()
        if not name:
            raise ValueError(f"Empty parameter name found in {config_path}")

        normalized_name = alias_map.get(name, name)
        if normalized_name in normalized:
            raise ValueError(
                f"Duplicate parameter {normalized_name!r} found inside the same config file: {config_path}"
            )
        normalized[normalized_name] = item["value"]

    return normalized


def parse_bool(value: Any) -> bool:
    """Parse a human-friendly boolean value from CLI or config input."""
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError("Boolean value cannot be empty")

    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def merge_config_dicts(
    base: Mapping[str, Any],
    script_specific: Mapping[str, Any],
    cli_args: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge configs and raise if base and script-specific share a key."""
    overlap = sorted(set(base).intersection(script_specific))
    if overlap:
        raise ValueError(f"Duplicate parameters found in config files: {overlap}")

    merged: Dict[str, Any] = dict(base)
    merged.update(script_specific)
    if cli_args:
        merged.update(cli_args)
    return merged


def apply_config_to_parser(parser: ArgumentParser, config_dict: Mapping[str, Any]) -> None:
    """Apply config values as parser defaults after validating the keys."""
    parser_dests = {action.dest for action in parser._actions if action.dest != "help"}
    unknown = sorted(set(config_dict).difference(parser_dests))
    if unknown:
        raise ValueError(f"Config contains unknown parameters for this parser: {unknown}")
    parser.set_defaults(**dict(config_dict))
