"""Arouse's reply formats, and the runtime's fallback answer.

The synthetic training data uses these formats for final answers. When the model's own
answer fails the answer guard (it uses words that are not in the conversation, i.e. an
invented fact) and no resample passes, the runtime answers from the last successful tool
result instead, in the same format. The facts then come from the tool, never from the model.
"""

from __future__ import annotations

from typing import Any

from arouse.agent.context import DAY_CODES, WEEKDAYS


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def rule_words(repeat: dict[str, Any]) -> str:
    f = repeat["freq"]
    if f == "daily":
        n = repeat.get("interval", 1)
        return "every day" if n == 1 else f"every {n} days"
    if f == "monthly":
        return f"on the {ordinal(repeat['by_month_day'][0])} of every month"
    days = repeat["by_day"]
    if days == list(DAY_CODES[:5]):
        return "every weekday"
    if days == ["SA", "SU"]:
        return "every weekend"
    names = [WEEKDAYS[DAY_CODES.index(d)].capitalize() for d in days]
    return "every " + " and ".join(names)


def created_msg(res: dict[str, Any], repeat: dict[str, Any] | None) -> str:
    date, time = res["next_run"].split("T")
    msg = f"Reminder set: {res['task']} on {date} at {time}"
    return msg + (f", repeating {rule_words(repeat)}." if repeat else ".")


def list_msg(items: list[dict[str, Any]]) -> str:
    if not items:
        return "You have no reminders."
    parts = [f"{x['task']} on {x['next_run'].replace('T', ' at ')}" + (" (repeats)" if "repeat" in x else "") for x in items]
    return f"You have {len(items)} reminder{'s' if len(items) > 1 else ''}: " + "; ".join(parts) + "."


def answer_from_result(events: list[dict[str, Any]]) -> str | None:
    """Final answer for the latest successful tool call in this request, or None."""
    for i in range(len(events) - 1, 0, -1):
        ev = events[i]
        if ev["type"] == "user":
            return None
        if ev["type"] == "tool_error":
            return None
        if ev["type"] == "tool_result":
            prev = events[i - 1]
            if prev["type"] != "arouse" or prev["turn"]["action"]["type"] != "tool_call":
                return None
            tool, args, res = prev["turn"]["action"]["tool"], prev["turn"]["action"]["arguments"], ev["content"]
            try:
                if tool == "scheduler.create":
                    return created_msg(res, args.get("repeat"))
                if tool == "scheduler.list":
                    return list_msg(res["reminders"])
                if tool == "scheduler.delete":
                    return f"Deleted the reminder: {res['task']}."
                if tool == "notes.create":
                    return "Saved the note."
                if tool == "file.read":
                    return f"{res['path']} has {res['lines']} lines. It starts with: {res['preview'].split(chr(10))[0]}"
                if tool == "file.list":
                    return f"You have {len(res['files'])} files: {', '.join(res['files'])}."
            except (KeyError, TypeError, AttributeError, ValueError):
                return None
            return None
    return None
