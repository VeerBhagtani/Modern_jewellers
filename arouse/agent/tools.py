"""Built-in tool specs and a deterministic sandbox that executes them.

The sandbox is what the local chat UI uses, and what the training-data generator
uses to produce tool results, so training data and runtime behaviour always agree.
MDA (or any other client) can instead execute tool calls itself via /v1/agent.
"""

from __future__ import annotations

import copy
import dataclasses
from datetime import datetime, timedelta
from typing import Any

from arouse.agent.context import DAY_CODES, fmt_dt
from arouse.protocol import Param, ProtocolError, ToolRegistry, ToolSpec

DATE = r"\d{4}-\d{2}-\d{2}"
TIME = r"([01]\d|2[0-3]):[0-5]\d"

REPEAT = Param("object", required=False, properties={
    "freq": Param("string", enum=("daily", "weekly", "monthly")),
    "interval": Param("integer", required=False, minimum=1, maximum=365),
    "by_day": Param("array", required=False, items=Param("string", enum=DAY_CODES)),
    "by_month_day": Param("array", required=False, items=Param("integer", minimum=1, maximum=28)),
})

TOOL_SPECS = [
    ToolSpec("scheduler.create", "Create a reminder: one-time (date+time, or time only = next occurrence), "
             "relative (in_minutes) or recurring (time+repeat).", {
        "task": Param("string"),
        "date": Param("string", required=False, pattern=DATE),
        "time": Param("string", required=False, pattern=TIME),
        "in_minutes": Param("integer", required=False, minimum=1, maximum=10080),
        "repeat": REPEAT,
    }),
    ToolSpec("scheduler.list", "List all reminders.", {}),
    ToolSpec("scheduler.delete", "Delete a reminder by task_id.", {"task_id": Param("string")}),
    ToolSpec("notes.create", "Save a note.", {"text": Param("string")}),
    ToolSpec("file.read", "Read a file.", {"path": Param("string")}),
    ToolSpec("file.list", "List files.", {}),
]
REGISTRY = ToolRegistry(TOOL_SPECS)


@dataclasses.dataclass
class Reminder:
    task_id: str
    task: str
    next_run: datetime
    repeat: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"task_id": self.task_id, "task": self.task, "next_run": fmt_dt(self.next_run)}
        if self.repeat:
            d["repeat"] = self.repeat
        return d


def first_run(now: datetime, hh: int, mm: int, repeat: dict[str, Any]) -> datetime:
    """Earliest time after `now` at hh:mm matching the recurrence rule."""
    freq = repeat["freq"]
    d = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    for _ in range(400):
        if d > now:
            if freq == "daily":
                return d
            if freq == "weekly" and DAY_CODES[d.weekday()] in repeat["by_day"]:
                return d
            if freq == "monthly" and d.day in repeat["by_month_day"]:
                return d
        d += timedelta(days=1)
    raise ProtocolError("repeat rule never matches")


def _err(message: str, retryable: bool = False) -> dict[str, Any]:
    return {"success": False, "error": message, "retryable": retryable}


class Sandbox:
    """In-memory tools. `failures` = upcoming transient failures per tool (for testing/data)."""

    def __init__(
        self,
        now: datetime,
        files: dict[str, str] | None = None,
        reminders: list[Reminder] | None = None,
        failures: dict[str, int] | None = None,
    ) -> None:
        self.now = now
        self.files = dict(files or {})
        self.reminders: dict[str, Reminder] = {r.task_id: r for r in (reminders or [])}
        self.notes: list[dict[str, str]] = []
        self.failures = dict(failures or {})
        self._next_id = 1 + max((int(r.task_id.split("-")[1]) for r in self.reminders.values()), default=0)

    def snapshot(self) -> dict[str, Any]:
        return {
            "reminders": [r.to_dict() for r in sorted(self.reminders.values(), key=lambda r: (r.next_run, r.task_id))],
            "notes": copy.deepcopy(self.notes),
        }

    def execute(self, tool: str, arguments: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Returns (ok, payload). Never raises for bad input: errors are results the agent must handle."""
        try:
            REGISTRY.validate_call(tool, arguments)
        except ProtocolError as e:
            return False, _err(f"invalid arguments: {e}")
        if self.failures.get(tool, 0) > 0:
            self.failures[tool] -= 1
            return False, _err(f"{tool.split('.')[0]} unavailable", retryable=True)
        handler = getattr(self, "_" + tool.replace(".", "_"))
        return handler(**arguments)

    # --- scheduler -----------------------------------------------------------

    def _scheduler_create(self, task: str, date: str | None = None, time: str | None = None,
                          in_minutes: int | None = None, repeat: dict[str, Any] | None = None) -> tuple[bool, dict]:
        if in_minutes is not None:
            if date or time or repeat:
                return False, _err("invalid arguments: in_minutes cannot be combined with date/time/repeat")
            when = self.now + timedelta(minutes=in_minutes)
        elif repeat is not None:
            if date or not time:
                return False, _err("invalid arguments: a recurring reminder needs time and repeat (no date)")
            f = repeat["freq"]
            if (f == "weekly") != ("by_day" in repeat) or (f == "monthly") != ("by_month_day" in repeat):
                return False, _err("invalid arguments: weekly needs by_day, monthly needs by_month_day")
            hh, mm = map(int, time.split(":"))
            when = first_run(self.now, hh, mm, repeat)
        elif time and not date:  # time only: the next occurrence of that clock time
            hh, mm = map(int, time.split(":"))
            when = first_run(self.now, hh, mm, {"freq": "daily"})
        else:
            if not (date and time):
                return False, _err("invalid arguments: need date and time, in_minutes, or time and repeat")
            try:
                when = datetime.strptime(f"{date}T{time}", "%Y-%m-%dT%H:%M")
            except ValueError:
                return False, _err(f"invalid date {date}")
            if when <= self.now:
                return False, _err("that time has already passed")
        rid = f"r-{self._next_id}"
        self._next_id += 1
        self.reminders[rid] = Reminder(rid, task, when, repeat)
        return True, {"success": True, "task_id": rid, "task": task, "next_run": fmt_dt(when)}

    def _scheduler_list(self) -> tuple[bool, dict]:
        items = self.snapshot()["reminders"]
        return True, {"success": True, "count": len(items), "reminders": items}

    def _scheduler_delete(self, task_id: str) -> tuple[bool, dict]:
        r = self.reminders.pop(task_id, None)
        if r is None:
            return False, _err(f"no reminder with task_id {task_id}")
        return True, {"success": True, "deleted": task_id, "task": r.task}

    # --- notes / files ---------------------------------------------------------

    def _notes_create(self, text: str) -> tuple[bool, dict]:
        nid = f"n-{len(self.notes) + 1}"
        self.notes.append({"note_id": nid, "text": text})
        return True, {"success": True, "note_id": nid}

    def _file_read(self, path: str) -> tuple[bool, dict]:
        if path not in self.files:
            return False, _err(f"file not found: {path}")
        content = self.files[path]
        lines = content.splitlines()
        preview = "\n".join(lines[:2])[:120]
        return True, {"success": True, "path": path, "lines": len(lines), "preview": preview}

    def _file_list(self) -> tuple[bool, dict]:
        return True, {"success": True, "files": sorted(self.files)}
