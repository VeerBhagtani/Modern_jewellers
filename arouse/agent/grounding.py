"""Grounding checks: free-text arguments must come from the conversation, not the model's
imagination. A reminder's task text must appear in what the user said in this request
and must end at a phrase boundary (so "renew the gym" is not accepted when the user
said "renew the gym membership"); a file path must be one the user named or a tool listed.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from arouse.agent.context import WEEKDAYS
from arouse.protocol import Action

_COPY_FIELDS = {"scheduler.create": "task", "notes.create": "text"}
_WORD = re.compile(r"[\w'’-]+|[^\w\s]")
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]
BOUNDARY_WORDS = {"at", "on", "in", "by", "after", "before", "from", "tomorrow", "today", "tonight", "every", "each",
                  "this", "next", "daily", "please", "pls", "thanks", "thank", "thx", "remind", "so", "then", "monthly",
                  "weekly", *WEEKDAYS,
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


_STOP = {"the", "a", "an", "to", "my", "for", "of", "on", "in", "at", "and", "me", "about", "reminder", "reminders", "it",
         "that", "this", "please", "i", "you"}


def _content(text: str) -> set[str]:
    return {w for w in _words(text) if w[0].isalnum() and w not in _STOP}


def delete_issue(task_id: str, events: list[dict[str, Any]]) -> str | None:
    """A reminder may only be deleted if a tool listed it in this request and its task is the
    listed reminder that best matches the user's words (in this request; if nothing in this
    request matches, e.g. "cancel that", in the whole conversation)."""
    req = current_request(events)
    listed: dict[str, str] = {}
    for ev in req:
        if ev["type"] == "tool_result":
            for r in ev["content"].get("reminders", []):
                listed[r["task_id"]] = r["task"]
            if "task_id" in ev["content"] and "task" in ev["content"]:
                listed[ev["content"]["task_id"]] = ev["content"]["task"]
    if task_id not in listed:
        return f"task_id {task_id} was not listed in this request"
    for scope in (req, events):
        said = set().union(*(_content(ev["content"]) for ev in scope if ev["type"] == "user"))
        score = {tid: len(_content(t) & said) / max(len(_content(t)), 1) for tid, t in listed.items()}
        if max(score.values()) > 0:
            if score[task_id] < max(score.values()):
                return f"'{listed[task_id]}' is not the reminder the user named"
            return None
    return "no listed reminder matches the user's words"


def grounding_issue(action: Action, events: list[dict[str, Any]]) -> str | None:
    """None if the action is grounded, else a short reason."""
    if action.type != "tool_call":
        return None
    if action.tool == "scheduler.delete":
        return delete_issue(str(action.arguments.get("task_id", "")), events)
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


_ANSWER_WORD = re.compile(r"[a-z0-9][a-z0-9_.:'-]*[a-z0-9]|[a-z0-9]", re.I)


def answer_words(text: str) -> list[str]:
    return [w.lower() for w in _ANSWER_WORD.findall(text)]


@lru_cache(maxsize=1)
def response_vocab() -> frozenset[str]:
    """Words Arouse uses in its own sentences (built by scripts/build_response_vocab.py)."""
    path = Path(__file__).with_name("response_vocab.txt")
    return frozenset(path.read_text(encoding="utf-8").split()) if path.exists() else frozenset()


def answer_issue(action: Action, events: list[dict[str, Any]]) -> str | None:
    """A final answer / question may only use Arouse's response vocabulary plus words that
    occur in this request's user messages or tool observations (no invented facts)."""
    text = action.result or action.question or action.error
    if not text or not response_vocab():
        return None
    sources = []
    for ev in current_request(events):
        if ev["type"] == "user":
            sources.append(ev["content"])
        elif ev["type"] in ("tool_result", "tool_error"):
            sources.append(json.dumps(ev["content"]))
        elif ev["type"] == "arouse" and ev["turn"]["action"]["type"] == "tool_call":
            sources.append(json.dumps(ev["turn"]["action"]["arguments"]))
    seen = {w for src in sources for w in answer_words(src)}
    blob = " ".join(sources).lower()
    vocab = response_vocab()
    unknown = [w for w in answer_words(text)
               if w not in vocab and w not in seen and not re.fullmatch(r"\d{1,3}(st|nd|rd|th)?", w)  # counts, ordinals
               and not (any(c.isdigit() for c in w) and w in blob)]
    if unknown:
        return f"answer uses words not in the conversation: {unknown[:5]}"
    return claim_issue(action, events)


_NOT_FOUND = re.compile(r"couldn't find a reminder to (.+?)\.?$", re.I)


def claim_issue(action: Action, events: list[dict[str, Any]]) -> str | None:
    """A "couldn't find a reminder to X" answer must agree with the tools (X was not listed)
    and must name what the user asked for (X is a phrase the user said)."""
    m = _NOT_FOUND.search(action.error or "") if action.type == "fail" else None
    if not m:
        return None
    target = m.group(1).strip()
    req = current_request(events)
    listed = {r["task"].lower() for ev in req if ev["type"] == "tool_result" for r in ev["content"].get("reminders", [])}
    if target.lower() in listed:
        return f"'{target}' was in the listed reminders"
    v = _words(target)
    if not any(_words(ev["content"])[i:i + len(v)] == v for ev in req if ev["type"] == "user"
               for i in range(len(_words(ev["content"])))):
        return f"'{target}' is not what the user asked for"
    return None
