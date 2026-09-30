"""Runtime context given to Arouse each turn.

Design rule: the model understands, deterministic software computes. Instead of
asking a small model to do calendar arithmetic, the runtime supplies the current
time and a calendar of upcoming dates; recurring and relative reminders are sent to
the scheduler as rules / offsets, and the scheduler computes the actual times.

The calendar is a plain string whose weekday names appear exactly as users type them
(" Friday 2026-10-02"), so resolving "on Friday" is a direct copy for the model.
Today's weekday is written in parentheses so "on Wednesday" (said on a Wednesday)
matches only next week's entry.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
DAY_CODES = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")


def fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M")


def next_weekday_date(now: datetime, weekday: int) -> datetime:
    """Next occurrence strictly after today ("Friday", "this Friday", "next Friday")."""
    days = (weekday - now.weekday()) % 7 or 7
    return now + timedelta(days=days)


def upcoming(now: datetime) -> dict[str, str]:
    """{"today", "tomorrow", "monday".."sunday"} -> YYYY-MM-DD."""
    out = {"today": now.strftime("%Y-%m-%d"), "tomorrow": (now + timedelta(days=1)).strftime("%Y-%m-%d")}
    for w in range(7):
        out[WEEKDAYS[w]] = next_weekday_date(now, w).strftime("%Y-%m-%d")
    return out


def calendar_text(now: datetime) -> str:
    u = upcoming(now)
    parts = [f"today {u['today']} ({WEEKDAYS[now.weekday()].capitalize()})", f"tomorrow {u['tomorrow']}"]
    for k in range(1, 8):
        d = now + timedelta(days=k)
        parts.append(f"{WEEKDAYS[d.weekday()].capitalize()} {d.strftime('%Y-%m-%d')}")
    return ", ".join(parts)


def build_context(now: datetime, timezone: str = "Asia/Kolkata") -> dict[str, Any]:
    return {"now": fmt_dt(now), "timezone": timezone, "calendar": calendar_text(now)}
