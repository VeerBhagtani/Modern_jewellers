"""What the user wrote: amounts, GST rates and clock times, exactly as written.

Structured values that Arouse puts in a tool call must come from the user's words (the
model chooses which one; it cannot invent one). Used by copy-constrained decoding and by
the grounding check. Mirrored in web/arouse.js.
"""

from __future__ import annotations

import re
from typing import Any

from arouse.agent.answers import Q_GST_RATE

_MONEY = re.compile(r"(?:₹\s?|rs\.?\s?|inr\s?)?[0-9][0-9,]*(?:\.[0-9]+)?(?:\s?(?:k|thousand|lakhs?|lacs?|crores?|cr)\b)?(?:\s?rupees)?", re.I)
_PERCENT_AFTER = re.compile(r"\s?(?:%|percent\b|per cent\b)", re.I)
_RATE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s?(?:%|percent\b|per cent\b)", re.I)
_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def money_and_rates(said: list[str], rate_replies: list[str]) -> tuple[set[str], set[str]]:
    """Amounts and GST rates exactly as the user wrote them. A number followed by % is a rate; in a
    reply to "Which GST rate should I use?" every number is a rate. Everything else is an amount."""
    amounts, rates = set(), set()
    for text in said:
        reply = text in rate_replies
        for m in _MONEY.finditer(text):
            span = m.group(0).rstrip(",").strip()
            if not reply and not _PERCENT_AFTER.match(text, m.end()) and span:
                amounts.add(span)
        rates |= {m.group(1) for m in _RATE.finditer(text)}
        if reply:
            rates |= set(_NUMBER.findall(text))
    return amounts, rates


_TIMES = [  # (pattern, how the hour is read)
    (re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s?(a\.?m\.?|p\.?m\.?)(?![a-z])", re.I), "ampm"),
    (re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s+in the (morning|afternoon|evening)\b", re.I), "period"),
    (re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s+at night\b", re.I), "night"),
    (re.compile(r"\b(\d{1,2}):(\d{2})\b(?!\s?(?:a\.?m|p\.?m))", re.I), "clock"),
    (re.compile(r"\bat (\d{1,2})\b(?![:.,]?\d|\s?(?:a\.?m|p\.?m|%)|\s+(?:in the|at night|minutes?|hours?|days?))", re.I), "bare"),
]


def mentioned_times(texts: list[str]) -> set[str]:
    """Clock times the user wrote, as HH:MM ("8 AM", "6:40pm", "9 in the morning", "20:30", "noon").
    A time without am/pm that could be either ("at 7", "8:30") allows both readings."""
    out = set()
    for text in texts:
        if re.search(r"\bnoon\b", text, re.I):
            out.add("12:00")
        if re.search(r"\bmidnight\b", text, re.I):
            out.add("00:00")
        taken: list[tuple[int, int]] = []  # earlier (more specific) patterns claim their text first
        for pat, kind in _TIMES:
            for m in pat.finditer(text):
                if any(m.start() < b and a < m.end() for a, b in taken):
                    continue
                taken.append((m.start(), m.end()))
                h, mm = int(m.group(1)), int(m.group(2) or 0) if kind != "bare" else 0
                if mm > 59 or h > 23:
                    continue
                if kind == "ampm":
                    if h == 0 or h > 12:
                        continue
                    h = h % 12 + (12 if m.group(3).lower().startswith("p") else 0)
                    cands = [h]
                elif kind == "period":
                    cands = [h % 12 + (0 if m.group(3).lower() == "morning" else 12)] if 1 <= h <= 12 else []
                elif kind == "night":
                    cands = [h % 12 + 12] if 7 <= h <= 11 else []
                elif kind == "clock" and (h >= 13 or m.group(1).startswith("0") or h == 0):
                    cands = [h]
                else:  # "8:30" or "at 7": morning or evening
                    cands = [h, h + 12] if 1 <= h <= 11 else [h] if h <= 23 else []
                out |= {f"{c:02d}:{mm:02d}" for c in cands if c <= 23}
    return out


def gst_mentions(request: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """(amounts, rates) the user wrote in this request."""
    said = [ev["content"] for ev in request if ev["type"] == "user"]
    replies = [nxt["content"] for ev, nxt in zip(request, request[1:])
               if ev["type"] == "arouse" and ev["turn"]["action"].get("question") == Q_GST_RATE and nxt["type"] == "user"]
    return money_and_rates(said, replies)
