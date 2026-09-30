#!/usr/bin/env python3
"""Build the generated part of datasets/tokenizer_tiny and its MANIFEST.json.

Deterministic: same seed -> byte-identical files (verified by tests).
Generated data is synthetic, template-based, written for Arouse; no external sources.

Usage:  python scripts/build_tiny_tokenizer_corpus.py [--out datasets/tokenizer_tiny] [--seed 1234]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from arouse.tokenizer.special_tokens import Special as S

HAND_WRITTEN = {
    "general.txt": "Original English prose (farm operations, planning, study, records, errors).",
    "code.txt": "Original Python / JavaScript / SQL / Bash snippets.",
    "structured.txt": "Original CSV / YAML / JSON / TOML samples, incl. a task-state object.",
    "multilingual.txt": "Original short reminder sentences in Hindi, Spanish, French, German.",
}

TASKS = [
    "check sales", "study geography", "call the feed supplier", "pay the electricity bill",
    "order mineral blocks", "update the milk records", "water the plants", "take my medicine",
    "send the weekly report", "clean the milking machine", "review the invoices", "book the vet visit",
    "back up the sales file", "call Ravi", "renew the insurance", "check the generator",
]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_CODES = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
SYSTEM = "You are Arouse, an agent that completes tasks by calling tools. Never claim success without a successful tool result."
TOOLS = [
    {"name": "scheduler.create", "arguments": {"task": "string", "date": "YYYY-MM-DD", "time": "HH:MM", "repeat": "object?"}},
    {"name": "file.read", "arguments": {"path": "string"}},
    {"name": "file.list", "arguments": {"dir": "string"}},
]


def js(obj: object) -> str:
    """Canonical compact JSON (the form the model will emit)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def seg(tok: S, body: str) -> str:
    return f"{tok.text}{body}{S.END.text}"


def fmt_time(h: int, m: int) -> str:
    return f"{h:02d}:{m:02d}"


def spoken_time(h: int, m: int) -> str:
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12} {suffix}" if m == 0 else f"{h12}:{m:02d} {suffix}"


def header(now: datetime) -> str:
    ctx = {"now": now.strftime("%Y-%m-%dT%H:%M"), "timezone": "Asia/Kolkata"}
    return S.BOS.text + seg(S.SYSTEM, SYSTEM) + seg(S.CONTEXT, js(ctx)) + seg(S.TOOLS, js(TOOLS))


def arouse(action: S, body: dict, plan: str | None = None, verify: str | None = None) -> str:
    out = S.AROUSE.text
    if plan:
        out += S.PLAN.text + plan
    if verify:
        out += S.VERIFY.text + verify
    return out + action.text + js(body) + S.END.text


def first_occurrence(now: datetime, h: int, m: int, ok=lambda d: True) -> datetime:
    """Earliest datetime at h:m that is after `now` and whose date satisfies `ok`."""
    d = now.replace(hour=h, minute=m, second=0, microsecond=0)
    while d <= now or not ok(d):
        d += timedelta(days=1)
    return d


def scenario(rng: random.Random) -> str:
    now = datetime(2026, 9, 1, 7, 0) + timedelta(minutes=rng.randrange(0, 60 * 24 * 60, 5))
    task = rng.choice(TASKS)
    h, m = rng.choice([6, 7, 8, 9, 10, 14, 17, 18, 20]), rng.choice([0, 0, 0, 15, 30, 45])
    kind = rng.choice(["tomorrow", "weekly", "ambiguous", "relative", "monthly", "every_n", "error_retry", "error_fail", "file_missing"])
    t = header(now)

    def ok(task_id: str) -> str:
        return seg(S.TOOL_RESULT, js({"success": True, "task_id": task_id}))

    tid = f"t-{rng.randrange(1000, 9999)}"
    if kind == "tomorrow":
        d = now + timedelta(days=1)
        t += seg(S.USER, f"Remind me tomorrow at {spoken_time(h, m)} to {task}.")
        args = {"task": task.capitalize(), "date": d.strftime("%Y-%m-%d"), "time": fmt_time(h, m)}
        t += arouse(S.TOOL_CALL, {"tool": "scheduler.create", "arguments": args}, plan="Create a one-time reminder for tomorrow.")
        t += ok(tid)
        t += arouse(S.FINISH, {"result": f"Reminder set for tomorrow at {fmt_time(h, m)}."}, verify=f"scheduler.create succeeded with task_id {tid}.")
    elif kind == "weekly":
        wd = rng.randrange(7)
        d = first_occurrence(now, h, m, lambda x: x.weekday() == wd)
        t += seg(S.USER, f"Every {WEEKDAYS[wd]} at {spoken_time(h, m)} remind me to {task}.")
        args = {"task": task.capitalize(), "date": d.strftime("%Y-%m-%d"), "time": fmt_time(h, m),
                "repeat": {"freq": "weekly", "interval": 1, "by_day": [DAY_CODES[wd]]}}
        t += arouse(S.TOOL_CALL, {"tool": "scheduler.create", "arguments": args}, plan="Create a weekly recurring reminder.")
        t += ok(tid)
        t += arouse(S.FINISH, {"result": f"Weekly reminder created for every {WEEKDAYS[wd]} at {fmt_time(h, m)}."}, verify=f"scheduler.create succeeded with task_id {tid}.")
    elif kind == "ambiguous":
        phrase = rng.choice(["tomorrow", f"next {rng.choice(WEEKDAYS)}", "every weekday morning", "every evening after dinner"])
        t += seg(S.USER, f"Remind me {phrase} to {task}.")
        t += arouse(S.ASK_USER, {"question": "What time should I remind you?"}, plan="The time is missing; ask instead of guessing.")
    elif kind == "relative":
        n = rng.choice([1, 2, 3, 4])
        d = now + timedelta(hours=n)
        t += seg(S.USER, f"In {n} hour{'s' if n > 1 else ''}, remind me to {task}.")
        args = {"task": task.capitalize(), "date": d.strftime("%Y-%m-%d"), "time": d.strftime("%H:%M")}
        t += arouse(S.TOOL_CALL, {"tool": "scheduler.create", "arguments": args}, plan=f"Now is {now:%H:%M}; add {n} hour(s).")
        t += ok(tid)
        t += arouse(S.FINISH, {"result": f"Reminder set for {d:%H:%M} today." if d.date() == now.date() else f"Reminder set for {d:%Y-%m-%d} at {d:%H:%M}."}, verify=f"scheduler.create succeeded with task_id {tid}.")
    elif kind == "monthly":
        first = first_occurrence(now, h, m, lambda x: x.day == 1)
        t += seg(S.USER, f"On the first day of every month at {spoken_time(h, m)}, remind me to {task}.")
        args = {"task": task.capitalize(), "date": first.strftime("%Y-%m-%d"), "time": fmt_time(h, m),
                "repeat": {"freq": "monthly", "interval": 1, "by_month_day": [1]}}
        t += arouse(S.TOOL_CALL, {"tool": "scheduler.create", "arguments": args}, plan="Create a monthly reminder on day 1.")
        t += ok(tid)
        t += arouse(S.FINISH, {"result": f"Monthly reminder created for the 1st at {fmt_time(h, m)}."}, verify=f"scheduler.create succeeded with task_id {tid}.")
    elif kind == "every_n":
        n = rng.choice([2, 3, 5])
        d = first_occurrence(now, h, m)
        t += seg(S.USER, f"Every {n} days at {spoken_time(h, m)} remind me to {task}.")
        args = {"task": task.capitalize(), "date": d.strftime("%Y-%m-%d"), "time": fmt_time(h, m),
                "repeat": {"freq": "daily", "interval": n}}
        t += arouse(S.TOOL_CALL, {"tool": "scheduler.create", "arguments": args}, plan=f"Create a reminder repeating every {n} days.")
        t += ok(tid)
        t += arouse(S.FINISH, {"result": f"Reminder created every {n} days at {fmt_time(h, m)}."}, verify=f"scheduler.create succeeded with task_id {tid}.")
    elif kind in ("error_retry", "error_fail"):
        d = now + timedelta(days=1)
        t += seg(S.USER, f"Remind me tomorrow at {spoken_time(h, m)} to {task}.")
        call = {"tool": "scheduler.create", "arguments": {"task": task.capitalize(), "date": d.strftime("%Y-%m-%d"), "time": fmt_time(h, m)}}
        err = seg(S.TOOL_ERROR, js({"success": False, "error": "scheduler unavailable", "retryable": True}))
        t += arouse(S.TOOL_CALL, call, plan="Create a one-time reminder for tomorrow.")
        t += err
        t += arouse(S.TOOL_CALL, call, verify="The call failed but is retryable; retry once.")
        if kind == "error_retry":
            t += ok(tid)
            t += arouse(S.FINISH, {"result": f"Reminder set for tomorrow at {fmt_time(h, m)}."}, verify=f"Retry succeeded with task_id {tid}.")
        else:
            t += err
            t += arouse(S.FAIL, {"error": "The scheduler is unavailable, so the reminder was not created. Please try again later."}, verify="Both attempts failed; the reminder does not exist.")
    else:  # file_missing
        name = rng.choice(["sales.csv", "invoices.csv", "milk_records.csv"])
        t += seg(S.USER, f"Read {name} and tell me how many rows it has.")
        t += arouse(S.TOOL_CALL, {"tool": "file.read", "arguments": {"path": name}}, plan="Read the file, then count rows.")
        t += seg(S.TOOL_ERROR, js({"success": False, "error": f"file not found: {name}", "retryable": False}))
        t += arouse(S.TOOL_CALL, {"tool": "file.list", "arguments": {"dir": "."}}, verify="The file is missing; list the directory to find it.")
        t += seg(S.TOOL_RESULT, js({"success": True, "files": ["notes.txt", "todo.txt"]}))
        t += arouse(S.ASK_USER, {"question": f"I could not find {name}. Where is it saved?"}, verify="No matching file in the directory.")
    return t + S.EOS.text


PHRASES = [
    "Remind me tomorrow at {t} to {task}.",
    "Remind me tomorrow to {task}.",
    "Every {wd} at {t} remind me to {task}.",
    "Every weekday morning remind me to {task}.",
    "After {n} hours remind me to {task}.",
    "On the first day of every month remind me to {task}.",
    "Remind me next {wd} to {task}.",
    "Every {n} days remind me to {task}.",
    "Every evening after dinner remind me to {task}.",
    "Set a reminder for {wd} at {t}: {task}.",
    "Can you remind me to {task} on {date} at {t24}?",
    "Schedule '{task}' for {date} {t24}, repeating weekly.",
    "Set a daily reminder at {t} to {task}.",
    "Don't let me forget to {task} this {wd}.",
]


def phrase(rng: random.Random) -> str:
    h, m = rng.randrange(5, 22), rng.choice([0, 15, 30, 45])
    date = (datetime(2026, 9, 1) + timedelta(days=rng.randrange(120))).strftime("%Y-%m-%d")
    return rng.choice(PHRASES).format(
        t=spoken_time(h, m), t24=fmt_time(h, m), task=rng.choice(TASKS), wd=rng.choice(WEEKDAYS),
        n=rng.choice([2, 3, 4, 6]), date=date,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out: Path, seed: int) -> None:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    gen = {
        "scheduling.jsonl": [phrase(rng) for _ in range(1000)],
        "agent_trajectories.jsonl": [scenario(rng) for _ in range(300)],
    }
    for name, docs in gen.items():
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
            for d in docs:
                fh.write(js({"text": d}) + "\n")

    files = []
    for name, desc in HAND_WRITTEN.items():
        files.append({"file": name, "origin": "hand-written for Arouse", "license": "project-owned",
                      "description": desc, "sha256": sha256(out / name)})
    for name in gen:
        files.append({"file": name, "origin": f"generated by scripts/build_tiny_tokenizer_corpus.py --seed {seed}",
                      "license": "project-owned", "description": "Synthetic, template-based.", "sha256": sha256(out / name)})
    manifest = {
        "dataset": "tokenizer_tiny",
        "purpose": "Tiny corpus for tokenizer development and tests. NOT for training a real tokenizer.",
        "external_sources": [],
        "files": files,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="datasets/tokenizer_tiny")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    build(Path(a.out), a.seed)


if __name__ == "__main__":
    main()
