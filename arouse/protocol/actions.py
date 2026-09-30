"""Arouse Action Protocol v1: the only way Arouse affects the world.

The model never executes anything. Each turn it emits exactly one action:

    {"type": "tool_call", "tool": "scheduler.create", "arguments": {...}}
    {"type": "ask_user",  "question": "What time should I remind you?"}
    {"type": "finish",    "result": "The reminder has been created."}
    {"type": "fail",      "error": "The scheduler is unavailable."}

Model-side encoding (one special token per action type, JSON body without "type"):

    <|arouse|> [<|plan|> text] [<|verify|> text] <|tool_call|>{"arguments":{...},"tool":"..."}<|end|>

JSON bodies are canonical: compact separators, UTF-8, nested keys sorted, and the
tool_call body always starts with "tool" (the model decides the tool before its
arguments). Deterministic.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

PROTOCOL_VERSION = "arouse-action/1"
ACTION_TYPES = ("tool_call", "ask_user", "finish", "fail")
TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
_REQUIRED = {"tool_call": ("tool", "arguments"), "ask_user": ("question",), "finish": ("result",), "fail": ("error",)}


class ProtocolError(ValueError):
    """An action or turn that violates the protocol."""


def _sorted(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sorted(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_sorted(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    """Compact JSON, keys sorted at every level except that a top-level "tool" key
    (tool_call bodies) always comes first."""
    obj = _sorted(obj)
    if isinstance(obj, dict) and "tool" in obj:
        obj = {"tool": obj["tool"], **{k: v for k, v in obj.items() if k != "tool"}}
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


@dataclasses.dataclass(frozen=True)
class Action:
    type: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    question: str | None = None
    result: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        validate_action(self.to_dict(validate=False))

    # constructors
    @staticmethod
    def tool_call(tool: str, arguments: dict[str, Any]) -> Action:
        return Action("tool_call", tool=tool, arguments=arguments)

    @staticmethod
    def ask_user(question: str) -> Action:
        return Action("ask_user", question=question)

    @staticmethod
    def finish(result: str) -> Action:
        return Action("finish", result=result)

    @staticmethod
    def fail(error: str) -> Action:
        return Action("fail", error=error)

    @property
    def is_terminal(self) -> bool:
        """The agent loop stops here and control returns to the user."""
        return self.type != "tool_call"

    def body(self) -> dict[str, Any]:
        """Fields after the action token (everything except "type")."""
        return {k: getattr(self, k) for k in _REQUIRED[self.type]}

    def to_dict(self, validate: bool = True) -> dict[str, Any]:
        d = {"type": self.type, **{k: getattr(self, k) for k in _REQUIRED.get(self.type, ())}}
        if validate:
            validate_action(d)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Action:
        validate_action(d)
        return Action(**d)


def validate_action(d: Any) -> None:
    if not isinstance(d, dict):
        raise ProtocolError("action must be an object")
    t = d.get("type")
    if t not in ACTION_TYPES:
        raise ProtocolError(f"type must be one of {ACTION_TYPES}, got {t!r}")
    allowed = {"type", *_REQUIRED[t]}
    extra = set(d) - allowed
    if extra:
        raise ProtocolError(f"{t}: unexpected fields {sorted(extra)}")
    missing = [k for k in _REQUIRED[t] if d.get(k) is None]
    if missing:
        raise ProtocolError(f"{t}: missing fields {missing}")
    if t == "tool_call":
        if not isinstance(d["tool"], str) or not TOOL_NAME_RE.match(d["tool"]):
            raise ProtocolError(f"invalid tool name {d['tool']!r}")
        if not isinstance(d["arguments"], dict):
            raise ProtocolError("arguments must be an object")
    else:
        key = _REQUIRED[t][0]
        if not isinstance(d[key], str) or not d[key].strip():
            raise ProtocolError(f"{t}: {key} must be a non-empty string")


@dataclasses.dataclass(frozen=True)
class Turn:
    """One Arouse turn: optional reasoning + exactly one action."""

    action: Action
    plan: str | None = None
    verify: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"action": self.action.to_dict()}
        if self.plan:
            out["plan"] = self.plan
        if self.verify:
            out["verify"] = self.verify
        return out
