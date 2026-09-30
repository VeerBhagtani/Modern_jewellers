"""Grounding checks: free-text arguments must come from the conversation, not the model's
imagination. A reminder's task text must appear in what the user said in this request
and must end at a phrase boundary (so "renew the gym" is not accepted when the user
said "renew the gym membership"); a file path must be one the user named or a tool listed.
"""

from __future__ import annotations

import re
from typing import Any

from arouse.agent.context import WEEKDAYS
from arouse.protocol import Action

_COPY_FIELDS = {"scheduler.create": "task", "notes.create": "text"}
_WORD = re.compile(r"[\w'’-]+|[^\w\s]")
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]
BOUNDARY_WORDS = {"at", "on", "in", "by", "after", "before", "from", "tomorrow", "today", "tonight", "every", "each",
                  "this", "next", "daily", "please", "remind", "so", "then", "monthly", "weekly", *WEEKDAYS,
                  *(m for m in _MONTHS), *(m[:3] for m in _MONTHS)}


def current_request(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Events since the last finish/fail (ask_user keeps the same request open)."""
    start = 0
    for i, ev in enumerate(events):
        if ev["type"] == "arouse" and ev["turn"]["action"]["type"] in ("finish", "fail"):
            start = i + 1
    return events[start:]


def _words(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def copied_from(value: str, messages: list[str]) -> bool:
    """`value` occurs as a whole phrase in one of the messages, followed by a boundary."""
    v = _words(value)
    if not v:
        return False
    for msg in messages:
        w = _words(msg)
        for i in range(len(w) - len(v) + 1):
            if w[i : i + len(v)] == v:
                nxt = w[i + len(v)] if i + len(v) < len(w) else None
                if nxt is None or not nxt[0].isalnum() or nxt in BOUNDARY_WORDS:
                    return True
    return False


def grounding_issue(action: Action, events: list[dict[str, Any]]) -> str | None:
    """None if the action is grounded, else a short reason."""
    if action.type != "tool_call":
        return None
    req = current_request(events)
    said = [ev["content"] for ev in req if ev["type"] == "user"]
    field = _COPY_FIELDS.get(action.tool)
    if field and field in action.arguments and not copied_from(action.arguments[field], said):
        return f"{field} '{action.arguments[field]}' is not a phrase the user said"
    if action.tool == "file.read":
        path = action.arguments.get("path", "")
        listed = {f for ev in req if ev["type"] == "tool_result" for f in ev["content"].get("files", [])}
        if path not in listed and not any(path.lower() in m.lower() for m in said):
            return f"path '{path}' was neither named by the user nor listed"
    return None
