"""Agent episodes: header + event list, and their exact token encoding.

The same functions render training data and runtime prompts (no train/serve skew).

Events (JSON-serialisable):
    {"type": "user", "content": "..."}
    {"type": "arouse", "turn": {"plan": "...", "verify": "...", "action": {...}}}
    {"type": "tool_result", "content": {...}}     successful tool output
    {"type": "tool_error",  "content": {...}}     failed tool output
"""

from __future__ import annotations

import dataclasses
from typing import Any

from arouse.protocol import Action, Turn, canonical_json, encode_turn_body
from arouse.protocol.codec import render_turn_text
from arouse.tokenizer import ArouseTokenizer, Special

SYSTEM_PROMPT = (
    "You are Arouse. Complete tasks with tools. Ask when a detail is missing. "
    "Never claim success without a successful tool result."
)
EVENT_TYPES = ("user", "arouse", "tool_result", "tool_error")


@dataclasses.dataclass
class Header:
    context: dict[str, Any]
    tools: list[str]
    system: str = SYSTEM_PROMPT
    memory: str | None = None  # retrieved long-term memory (runtime decides what is relevant)
    state: dict[str, Any] | None = None  # task state for resuming


def turn_from_event(ev: dict[str, Any]) -> Turn:
    t = ev["turn"]
    return Turn(Action.from_dict(t["action"]), plan=t.get("plan"), verify=t.get("verify"))


def turn_event(turn: Turn) -> dict[str, Any]:
    return {"type": "arouse", "turn": turn.to_dict()}


def validate_events(events: list[dict[str, Any]]) -> None:
    for i, ev in enumerate(events):
        if not isinstance(ev, dict) or ev.get("type") not in EVENT_TYPES:
            raise ValueError(f"events[{i}]: type must be one of {EVENT_TYPES}")
        if ev["type"] == "user" and not isinstance(ev.get("content"), str):
            raise ValueError(f"events[{i}]: user content must be a string")
        if ev["type"] in ("tool_result", "tool_error") and not isinstance(ev.get("content"), dict):
            raise ValueError(f"events[{i}]: tool content must be an object")
        if ev["type"] == "arouse":
            turn_from_event(ev)  # raises ProtocolError if invalid


def _segment(tok: ArouseTokenizer, marker: Special, text: str) -> list[int]:
    return [marker, *tok.encode(text), Special.END]


def encode_header(tok: ArouseTokenizer, h: Header) -> list[int]:
    ids = [Special.BOS, *_segment(tok, Special.SYSTEM, h.system)]
    ids += _segment(tok, Special.CONTEXT, canonical_json(h.context))
    ids += _segment(tok, Special.TOOLS, canonical_json(h.tools))
    if h.memory:
        ids += _segment(tok, Special.MEMORY, h.memory)
    if h.state:
        ids += _segment(tok, Special.STATE, canonical_json(h.state))
    return ids


def encode_event(tok: ArouseTokenizer, ev: dict[str, Any]) -> list[int]:
    t = ev["type"]
    if t == "user":
        return _segment(tok, Special.USER, ev["content"])
    if t == "arouse":
        return [Special.AROUSE, *encode_turn_body(tok, turn_from_event(ev))]
    marker = Special.TOOL_RESULT if t == "tool_result" else Special.TOOL_ERROR
    return _segment(tok, marker, canonical_json(ev["content"]))


def encode_prompt(tok: ArouseTokenizer, h: Header, events: list[dict[str, Any]]) -> list[int]:
    """Prompt for the next Arouse turn (ends with <|arouse|>)."""
    ids = encode_header(tok, h)
    for ev in events:
        ids += encode_event(tok, ev)
    return ids + [Special.AROUSE]


def render_text(h: Header, events: list[dict[str, Any]], *, complete: bool = True) -> str:
    """Text form of an episode (training data). Equal to the token encoding for trusted content."""
    out = Special.BOS.text
    out += f"{Special.SYSTEM.text}{h.system}{Special.END.text}"
    out += f"{Special.CONTEXT.text}{canonical_json(h.context)}{Special.END.text}"
    out += f"{Special.TOOLS.text}{canonical_json(h.tools)}{Special.END.text}"
    if h.memory:
        out += f"{Special.MEMORY.text}{h.memory}{Special.END.text}"
    if h.state:
        out += f"{Special.STATE.text}{canonical_json(h.state)}{Special.END.text}"
    for ev in events:
        t = ev["type"]
        if t == "user":
            out += f"{Special.USER.text}{ev['content']}{Special.END.text}"
        elif t == "arouse":
            out += render_turn_text(turn_from_event(ev))
        else:
            marker = Special.TOOL_RESULT if t == "tool_result" else Special.TOOL_ERROR
            out += f"{marker.text}{canonical_json(ev['content'])}{Special.END.text}"
    return out + (Special.EOS.text if complete else "")
