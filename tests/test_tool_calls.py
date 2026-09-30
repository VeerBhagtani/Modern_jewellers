"""Sandbox tool execution: the deterministic side of every tool call."""

from datetime import datetime

import pytest

from arouse.agent.tools import REGISTRY, Reminder, Sandbox, first_run

NOW = datetime(2026, 9, 30, 19, 21)  # a Wednesday


def sb(**kw):
    return Sandbox(NOW, files={"sales.csv": "date,qty\n2026-09-01,4\n2026-09-02,5", "notes.txt": "hello"}, **kw)


def test_registry_names():
    assert REGISTRY.names() == ["scheduler.create", "scheduler.list", "scheduler.delete", "notes.create", "file.read", "file.list"]


def test_one_time_create():
    s = sb()
    ok, r = s.execute("scheduler.create", {"task": "Check sales", "date": "2026-10-01", "time": "08:00"})
    assert ok and r == {"success": True, "task_id": "r-1", "task": "Check sales", "next_run": "2026-10-01T08:00"}


def test_time_only_means_next_occurrence():
    ok, r = sb().execute("scheduler.create", {"task": "a", "time": "20:00"})
    assert r["next_run"] == "2026-09-30T20:00"
    ok, r = sb().execute("scheduler.create", {"task": "a", "time": "07:00"})
    assert r["next_run"] == "2026-10-01T07:00"


def test_relative_create():
    ok, r = sb().execute("scheduler.create", {"task": "a", "in_minutes": 120})
    assert ok and r["next_run"] == "2026-09-30T21:21"


@pytest.mark.parametrize(
    "repeat,expected",
    [
        ({"freq": "weekly", "by_day": ["MO"]}, "2026-10-05T09:00"),
        ({"freq": "weekly", "by_day": ["WE"]}, "2026-10-07T09:00"),  # today's 09:00 already passed
        ({"freq": "weekly", "by_day": ["MO", "TU", "WE", "TH", "FR"]}, "2026-10-01T09:00"),
        ({"freq": "daily"}, "2026-10-01T09:00"),
        ({"freq": "daily", "interval": 3}, "2026-10-01T09:00"),
        ({"freq": "monthly", "by_month_day": [1]}, "2026-10-01T09:00"),
        ({"freq": "monthly", "by_month_day": [15]}, "2026-10-15T09:00"),
    ],
)
def test_recurring_first_run(repeat, expected):
    ok, r = sb().execute("scheduler.create", {"task": "a", "time": "09:00", "repeat": repeat})
    assert ok and r["next_run"] == expected


def test_first_run_same_day_later():
    assert first_run(NOW, 21, 0, {"freq": "weekly", "by_day": ["WE"]}) == datetime(2026, 9, 30, 21, 0)


@pytest.mark.parametrize(
    "args,msg",
    [
        ({"task": "a", "date": "2026-09-30", "time": "08:00"}, "already passed"),
        ({"task": "a", "date": "2026-02-30", "time": "08:00"}, "invalid date"),
        ({"task": "a"}, "need date and time"),
        ({"task": "a", "date": "2026-10-01"}, "need date and time"),
        ({"task": "a", "in_minutes": 5, "time": "08:00"}, "cannot be combined"),
        ({"task": "a", "date": "2026-10-01", "time": "08:00", "repeat": {"freq": "daily"}}, "no date"),
        ({"task": "a", "time": "08:00", "repeat": {"freq": "weekly"}}, "by_day"),
        ({"task": "a", "time": "25:00"}, "invalid arguments"),
        ({"task": "a", "in_minutes": 0}, "invalid arguments"),
        ({"task": "", "time": "08:00"}, "invalid arguments"),
        ({"task": "a", "time": "08:00", "colour": "red"}, "invalid arguments"),
    ],
)
def test_create_errors_are_results_not_exceptions(args, msg):
    s = sb()
    ok, r = s.execute("scheduler.create", args)
    assert not ok and r["success"] is False and msg in r["error"] and r["retryable"] is False
    assert s.snapshot()["reminders"] == []  # nothing was created


def test_list_and_delete():
    s = sb(reminders=[Reminder("r-7", "Pay rent", datetime(2026, 10, 1, 10, 0))])
    ok, r = s.execute("scheduler.create", {"task": "b", "time": "20:00"})
    assert r["task_id"] == "r-8"  # ids continue after existing reminders
    ok, lst = s.execute("scheduler.list", {})
    assert lst["count"] == 2 and [x["task_id"] for x in lst["reminders"]] == ["r-8", "r-7"]  # sorted by time
    ok, d = s.execute("scheduler.delete", {"task_id": "r-7"})
    assert ok and d == {"success": True, "deleted": "r-7", "task": "Pay rent"}
    ok, d = s.execute("scheduler.delete", {"task_id": "r-7"})
    assert not ok and "no reminder" in d["error"]


def test_notes_and_files():
    s = sb()
    ok, n = s.execute("notes.create", {"text": "Vet on Tuesday"})
    assert ok and n["note_id"] == "n-1" and s.snapshot()["notes"] == [{"note_id": "n-1", "text": "Vet on Tuesday"}]
    ok, f = s.execute("file.read", {"path": "sales.csv"})
    assert ok and f == {"success": True, "path": "sales.csv", "lines": 3, "preview": "date,qty\n2026-09-01,4"}
    ok, f = s.execute("file.read", {"path": "../etc/passwd"})
    assert not ok and "not found" in f["error"]  # sandbox: only its own files exist
    ok, f = s.execute("file.list", {})
    assert f["files"] == ["notes.txt", "sales.csv"]


def test_unknown_tool_and_injected_failures():
    s = sb(failures={"scheduler.create": 1})
    ok, r = s.execute("shell.run", {"cmd": "rm -rf /"})
    assert not ok and "unknown tool" in r["error"]
    ok, r = s.execute("scheduler.create", {"task": "a", "time": "20:00"})
    assert not ok and r["retryable"] is True
    ok, r = s.execute("scheduler.create", {"task": "a", "time": "20:00"})
    assert ok
