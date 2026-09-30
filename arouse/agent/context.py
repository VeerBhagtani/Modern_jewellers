"""Runtime context given to Arouse each turn.

Design rule: the model understands, deterministic software computes. Instead of
asking a small model to do calendar arithmetic, the runtime supplies the current
time and a lookup of upcoming dates; recurring and relative reminders are sent to
the scheduler as rules / offsets, and the scheduler computes the actual times.
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


def build_context(now: datetime, timezone: str = "Asia/Kolkata") -> dict[str, Any]:
    return {
        "now": fmt_dt(now),
        "weekday": WEEKDAYS[now.weekday()],
        "timezone": timezone,
        "tomorrow": (now + timedelta(days=1)).strftime("%Y-%m-%d"),
        "next": {WEEKDAYS[w][:3]: next_weekday_date(now, w).strftime("%Y-%m-%d") for w in range(7)},
    }
