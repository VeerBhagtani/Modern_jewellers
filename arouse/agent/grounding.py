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
from arouse.agent.mentions import gst_mentions, mentioned_times
from arouse.protocol import Action

_COPY_FIELDS = {"scheduler.create": "task", "notes.create": "text", "kb.search": "query", "leads.find": "query"}
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


def _names_file(msg: str, path: str) -> bool:
    """`path` occurs in `msg` as a whole name ("payments.c" is not named by "payments.csv")."""
    low, p = msg.lower(), path.lower()
    i = low.find(p)
    while p and i != -1:
        after = low[i + len(p):i + len(p) + 1]
        if not after or not (after.isalnum() or after == "_"):
            return True
        i = low.find(p, i + 1)
    return False


def grounding_issue(action: Action, events: list[dict[str, Any]]) -> str | None:
    """None if the action is grounded, else a short reason."""
    if action.type != "tool_call":
        return None
    if action.tool == "scheduler.delete":
        return delete_issue(str(action.arguments.get("task_id", "")), events)
    req = current_request(events)
    for i in range(len(req) - 1):  # repeating a call that failed for good can never help
        prev, obs = req[i], req[i + 1]
        if (prev["type"] == "arouse" and obs["type"] == "tool_error" and not obs["content"].get("retryable")
                and prev["turn"]["action"].get("tool") == action.tool and prev["turn"]["action"].get("arguments") == action.arguments):
            return f"this exact call already failed: {obs['content'].get('error')}"
    said = [ev["content"] for ev in req if ev["type"] == "user"]
    field = _COPY_FIELDS.get(action.tool)
    if field and field in action.arguments and not copied_from(action.arguments[field], said):
        return f"{field} '{action.arguments[field]}' is not a phrase the user said"
    if action.tool == "gst.calculate":  # an amount and a rate the user wrote (a rate is written with %)
        amounts, rates = gst_mentions(req)
        for key, allowed in (("amount", amounts), ("rate", rates)):
            v = str(action.arguments.get(key, "")).strip()
            if (allowed and v.lower() not in {x.lower() for x in allowed}) or not v or not any(v.lower() in m.lower() for m in said):
                return f"{key} '{v}' is not the {key} the user wrote"
    if action.tool == "scheduler.create" and "time" in action.arguments:
        times = mentioned_times(said)
        if times and action.arguments["time"] not in times:
            return f"time {action.arguments['time']} is not a time the user wrote ({sorted(times)})"
    if action.tool == "file.read":
        path = action.arguments.get("path", "")
        listed = {f for ev in req if ev["type"] == "tool_result" for f in ev["content"].get("files", [])}
        if path not in listed and not any(_names_file(m, path) for m in said):
            return f"path '{path}' was neither named by the user nor listed"
    return None


_ANSWER_WORD = re.compile(r"[a-z0-9][a-z0-9_.:'-]*[a-z0-9]|[a-z0-9]", re.I)


_GROUPED = re.compile(r"(?<=\d),(?=\d)")
_TRAILING_ZEROS = re.compile(r"(\d+)\.(\d*?)0+(?!\d)")


def normalize_numbers(text: str) -> str:
    """'₹1,374.10' and the tool's 1374.1 must compare equal: drop digit grouping and trailing decimal zeros."""
    text = _GROUPED.sub("", text)
    return _TRAILING_ZEROS.sub(lambda m: m.group(1) + ("." + m.group(2) if m.group(2) else ""), text)


def answer_words(text: str) -> list[str]:
    return [w.lower() for w in _ANSWER_WORD.findall(normalize_numbers(text))]


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
    blob = normalize_numbers(" ".join(sources)).lower()
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
