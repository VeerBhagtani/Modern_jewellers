"""Tool specifications and argument validation (small, dependency-free schema)."""

from __future__ import annotations

import dataclasses
import re
from typing import Any

from arouse.protocol.actions import TOOL_NAME_RE, ProtocolError

_TYPES = {"string": str, "integer": int, "object": dict, "array": list, "boolean": bool}


@dataclasses.dataclass(frozen=True)
class Param:
    type: str
    required: bool = True
    description: str = ""
    pattern: str | None = None  # strings: full-match regex
    enum: tuple[Any, ...] | None = None
    minimum: int | None = None  # integers
    maximum: int | None = None
    items: Param | None = None  # arrays
    properties: dict[str, Param] | None = None  # objects (None = any keys)

    def check(self, value: Any, path: str) -> None:
        py = _TYPES[self.type]
        if not isinstance(value, py) or (py is int and isinstance(value, bool)):
            raise ProtocolError(f"{path}: expected {self.type}")
        if self.enum is not None and value not in self.enum:
            raise ProtocolError(f"{path}: must be one of {list(self.enum)}")
        if self.type == "string":
            if not value.strip():
                raise ProtocolError(f"{path}: must not be empty")
            if self.pattern and not re.fullmatch(self.pattern, value):
                raise ProtocolError(f"{path}: invalid format")
        if self.type == "integer":
            if self.minimum is not None and value < self.minimum:
                raise ProtocolError(f"{path}: must be >= {self.minimum}")
            if self.maximum is not None and value > self.maximum:
                raise ProtocolError(f"{path}: must be <= {self.maximum}")
        if self.type == "array" and self.items is not None:
            if not value:
                raise ProtocolError(f"{path}: must not be empty")
            for i, v in enumerate(value):
                self.items.check(v, f"{path}[{i}]")
        if self.type == "object" and self.properties is not None:
            check_object(self.properties, value, path)


def check_object(props: dict[str, Param], value: dict[str, Any], path: str) -> None:
    extra = set(value) - set(props)
    if extra:
        raise ProtocolError(f"{path}: unexpected {sorted(extra)}")
    for name, p in props.items():
        if name in value:
            p.check(value[name], f"{path}.{name}")
        elif p.required:
            raise ProtocolError(f"{path}: missing {name}")


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Param]

    def __post_init__(self) -> None:
        if not TOOL_NAME_RE.match(self.name):
            raise ValueError(f"invalid tool name {self.name!r}")

    def validate(self, arguments: dict[str, Any]) -> None:
        check_object(self.parameters, arguments, self.name)

    def describe(self) -> dict[str, Any]:
        def p(x: Param) -> dict[str, Any]:
            d: dict[str, Any] = {"type": x.type, "required": x.required}
            for k in ("description", "pattern", "minimum", "maximum"):
                if getattr(x, k) not in (None, ""):
                    d[k] = getattr(x, k)
            if x.enum:
                d["enum"] = list(x.enum)
            if x.items:
                d["items"] = p(x.items)
            if x.properties:
                d["properties"] = {k: p(v) for k, v in x.properties.items()}
            return d

        return {"name": self.name, "description": self.description, "parameters": {k: p(v) for k, v in self.parameters.items()}}


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]) -> None:
        self.specs = {s.name: s for s in specs}

    def names(self) -> list[str]:
        return list(self.specs)

    def validate_call(self, tool: str, arguments: dict[str, Any]) -> None:
        if tool not in self.specs:
            raise ProtocolError(f"unknown tool {tool!r}")
        self.specs[tool].validate(arguments)
