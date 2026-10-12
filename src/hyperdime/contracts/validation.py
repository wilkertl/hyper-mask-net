"""Error types and validation entry points shared by records and configs (research R1, R2)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

if TYPE_CHECKING:
    from hyperdime.contracts.config import RunConfig


class ContractError(ValueError):
    """An invalid record, manifest, or artifact; the only error type for contract violations."""


class ConfigError(ContractError):
    """An invalid run configuration."""


def format_validation_error(
    model_name: str, error: ValidationError, line: int | None = None
) -> str:
    """One ``<model>.<dotted.field>: <message>`` line per error, prefixed for JSON Lines."""
    prefix = f"line {line}: " if line is not None else ""
    lines = []
    for item in error.errors():
        field = ".".join(str(part) for part in item["loc"])
        location = f"{model_name}.{field}" if field else model_name
        lines.append(f"{prefix}{location}: {item['msg']}")
    return "\n".join(lines)


def format_config_error(error: ValidationError) -> str:
    """One ``<dotted.field>: <message>`` line per error, with the offending value when scalar."""
    lines = []
    for item in error.errors():
        field = ".".join(str(part) for part in item["loc"]) or "<root>"
        value = item.get("input")
        got = f" (got {value!r})" if isinstance(value, str | int | float | bool) else ""
        lines.append(f"{field}: {item['msg']}{got}")
    return "\n".join(lines)


def validate_config(raw: Mapping[str, Any]) -> RunConfig:
    """Validate a composed config into a ``RunConfig``; raises ``ConfigError``."""
    from hyperdime.contracts.config import validate_config as validate

    return validate(raw)
