"""Strict, typed configuration base shared by every Arouse config.

Every config (model, tokenizer, later training/inference) is a dataclass that:
  * type-checks its fields on construction,
  * rejects unknown keys (typos in YAML fail loudly instead of being ignored),
  * validates cross-field constraints in `validate()`,
  * round-trips through plain dicts and YAML.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T", bound="ConfigBase")


class ConfigError(ValueError):
    """Raised for any invalid configuration."""


def _matches(value: Any, hint: Any) -> bool:
    if hint is Any:
        return True
    if hint is type(None):
        return value is None
    origin = get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        return any(_matches(value, arg) for arg in get_args(hint))
    if origin is list:
        (item,) = get_args(hint) or (Any,)
        return isinstance(value, list) and all(_matches(v, item) for v in value)
    if origin is dict:
        key, val = get_args(hint) or (Any, Any)
        return isinstance(value, dict) and all(
            _matches(k, key) and _matches(v, val) for k, v in value.items()
        )
    if hint is bool:
        return isinstance(value, bool)
    if hint is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if hint is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, hint)


@dataclasses.dataclass
class ConfigBase:
    def __post_init__(self) -> None:
        hints = get_type_hints(type(self))
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            hint = hints[f.name]
            if not _matches(value, hint):
                raise ConfigError(
                    f"{type(self).__name__}.{f.name}: expected {hint}, got {value!r}"
                )
            # YAML writes 10000 for 10000.0; normalise ints in float fields.
            if hint is float and isinstance(value, int):
                setattr(self, f.name, float(value))
        self.validate()

    def validate(self) -> None:
        """Cross-field checks. Subclasses override; raise ConfigError on failure."""

    # --- dict / YAML IO -------------------------------------------------

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        if not isinstance(data, dict):
            raise ConfigError(f"{cls.__name__}: expected a mapping, got {type(data).__name__}")
        known = {f.name for f in dataclasses.fields(cls) if f.init}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"{cls.__name__}: unknown keys {unknown}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_yaml(cls: type[T], path: str | Path) -> T:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data or {})

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def replace(self: T, **changes: Any) -> T:
        """Copy with overrides; re-runs type checks and validation."""
        return dataclasses.replace(self, **changes)
