"""Copy-constrained decoding for free-text arguments.

While the model is writing the value of a copy field ("task", "text", "path") inside a
tool call, the only tokens allowed are those that continue a phrase from the allowed
sources (the user's words in this request; for paths also file names returned by a
tool), plus the closing quote once the value is a complete phrase ending at a boundary.
The model still chooses which phrase to copy and where it ends; it cannot invent words.

"date" values are restricted to the dates this request refers to: "today"/"tonight",
"tomorrow", weekday names (their next occurrence, as in the runtime calendar) and explicit
dates ("October 5", "5th of October"), read from the user's messages and Arouse's own
questions (e.g. "...should I set it for tomorrow instead?"). If the request names no day,
any calendar date is allowed. The model still chooses which one.
Everything else in the turn is decoded unconstrained.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

import torch

from arouse.agent.context import WEEKDAYS, next_weekday_date
from arouse.agent.grounding import copied_from, current_request
from arouse.agent.mentions import gst_mentions, mentioned_times
from arouse.tokenizer import ArouseTokenizer, Special

_OPEN = re.compile(r'"(task|text|query|path|date|time|amount|rate)":"((?:[^"\\]|\\.)*)$')
_CHOICE_FIELDS = ("date", "time", "amount", "rate")  # values chosen from a fixed list of candidates
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]
_MONTH_RE = "|".join(sorted({*_MONTHS, *(m[:3] for m in _MONTHS), "sept"}, key=len, reverse=True))
_DATE_PATTERNS = [
    re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.I),  # October 5
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_RE})\b", re.I),  # 5 October / 5th of October
]
# Where a copied value must stop: sentence punctuation (all fields) and, for reminder
# tasks, the start of a time/date phrase (" at 5", " in 3 hours", " on Sunday", " every ...").
_STOP_ALL = re.compile(r"[?!;]|\.(?=\s|$)|\s(?:please|pls|thanks|thank you|thx)\b", re.I)  # "4.50" is not an end
_STOP_PATH = re.compile(r"[\s?!;,]|\.(?:\s|$)")  # a file name ends at whitespace or a sentence-final "."
_DAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_STOP_TASK = re.compile(
    rf"[,:]|\s(?:at|in|on|by|after|from|before)\s+(?:\d|an?\s|half|the\s\d|{_DAYS}|noon|midnight)"
    rf"|\s(?:tomorrow|today|tonight)\b|\s(?:every|each)\s|\s(?:{_DAYS})\b"
    # "daily"/"weekly"/"monthly" end a task only as a time phrase ("... daily at 9"), not in "the weekly report"
    r"|\s(?:daily|weekly|monthly)(?=\s*(?:$|[.,!?;]|(?:at|on|in|from|starting|please|thanks)\b))"
    rf"|\s(?:next|this)\s+(?:{_DAYS}|week|weekend|month|year|morning|afternoon|evening|night)\b",
    re.I,
)
_CLOSE = re.compile(r'^"([,}\]].*)?$', re.S)  # closes the string; the rest is ordinary JSON structure
_VOCAB: dict[str, tuple[dict[str, list[int]], list[int], int]] = {}


_LABEL = re.compile(r"^\s*([^:\n]{1,60}?):\s+(?=\S)")


def label_end(src: str) -> int | None:
    """For "<short label>: <content>" ("Please jot down: ...", "Memo: ..."), where the content starts.
    Up to two labels are skipped ("quick one: note this down: ...")."""
    end = None
    for _ in range(2):
        m = _LABEL.match(src[end or 0:])
        if not m or len(m.group(1).split()) > 6:
            break
        end = (end or 0) + m.end()
    return end


def _ends_sentence(rest: str) -> bool:
    """`rest` (what follows a copied note) starts at the end of its sentence."""
    m = _STOP_ALL.search(rest)
    before = rest if m is None else rest[:m.start()]
    return not any(c.isalnum() for c in before)  # only spaces/punctuation before the stop (", thanks")


def mentioned_dates(texts: list[str], today: date) -> set[str]:
    """Next occurrence (on or after today) of each explicit month/day the user wrote."""
    out = set()
    for text in texts:
        for i, pat in enumerate(_DATE_PATTERNS):
            for m in pat.finditer(text):
                month_s, day_s = (m.group(1), m.group(2)) if i == 0 else (m.group(2), m.group(1))
                month = next(k for k, name in enumerate(_MONTHS, 1) if name.startswith(month_s.lower()[:3]))
                for year in (today.year, today.year + 1):
                    try:
                        d = date(year, month, int(day_s))
                    except ValueError:
                        break
                    if d >= today:
                        out.add(d.isoformat())
                        break
    return out


def referenced_dates(texts: list[str], now: datetime) -> set[str]:
    """Dates the texts refer to (see the module docstring)."""
    today = now.date()
    out = set()
    for text in texts:
        low = text.lower()
        words = set(re.findall(r"[a-z]+", low))
        if words & {"today", "tonight"}:
            out.add(today.isoformat())
        if "tomorrow" in words:
            out.add((today + timedelta(days=2 if "after tomorrow" in low else 1)).isoformat())
        for w, name in enumerate(WEEKDAYS):
            if name in words:
                out.add(next_weekday_date(now, w).date().isoformat())
    return out | mentioned_dates(texts, today)


def calendar_dates(context: dict[str, Any] | None) -> set[str]:
    """Every YYYY-MM-DD the runtime context mentions (today, tomorrow, upcoming weekdays)."""
    if not context or "now" not in context:
        return set()
    return {context["now"][:10], *re.findall(r"\d{4}-\d{2}-\d{2}", str(context.get("calendar", "")))}


def _vocab(tok: ArouseTokenizer) -> tuple[dict[str, list[int]], list[int], int]:
    """piece -> token ids (for pieces usable inside a JSON string), closing-quote tokens, max piece length."""
    key = tok.fingerprint()
    if key not in _VOCAB:
        pieces: dict[str, list[int]] = {}
        closing = []
        for t in range(tok.vocab_size):
            if tok.is_special(t):
                continue
            try:
                p = tok.id_to_bytes(t).decode("utf-8")
            except UnicodeDecodeError:
                continue
            if _CLOSE.match(p):
                closing.append(t)
            elif p and '"' not in p and "\\" not in p:
                pieces.setdefault(p, []).append(t)
        _VOCAB[key] = (pieces, closing, max(map(len, pieces)))
    return _VOCAB[key]


class CopyConstraint:
    def __init__(self, tok: ArouseTokenizer, events: list[dict[str, Any]], context: dict[str, Any] | None = None) -> None:
        self.tok = tok
        self.pieces, self.closing, self.maxlen = _vocab(tok)
        self.closer_text = {t: tok.id_to_bytes(t).decode("utf-8") for t in self.closing}
        req = current_request(events)
        said = [ev["content"] for ev in req if ev["type"] == "user"]
        listed = [f for ev in req if ev["type"] == "tool_result" for f in ev["content"].get("files", [])]
        self.sources = {"task": said, "text": said, "query": said, "path": said + listed}
        self.listed = set(listed)
        dates = calendar_dates(context)
        if dates:
            asked = [ev["turn"]["action"]["question"] for ev in req
                     if ev["type"] == "arouse" and ev["turn"]["action"]["type"] == "ask_user"]
            dates = referenced_dates(said + asked, datetime.strptime(context["now"][:16], "%Y-%m-%dT%H:%M")) or dates
        self.dates = sorted(dates)
        amounts, rates = gst_mentions(req)
        self.path_spans = set()
        for src in said:  # "vet_visits.txt" in "Count the lines in vet_visits.txt." (never "vet_visits.")
            for i, ch in enumerate(src):
                if ch.isalnum() and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] in "_-./\\")):
                    m = _STOP_PATH.search(src[i:])
                    span = src[i:i + m.start()] if m else src[i:]
                    if "." in span:
                        self.path_spans.add(span)
        self.choices = {"date": self.dates, "time": sorted(mentioned_times(said)), "amount": sorted(amounts), "rate": sorted(rates)}

    def open_field(self, generated: list[int]) -> tuple[str, str, str] | None:
        """(field, partial value, body so far) if the model is inside a constrained string."""
        if Special.TOOL_CALL not in generated:
            return None
        start = len(generated) - 1 - generated[::-1].index(Special.TOOL_CALL)
        body = b"".join(self.tok.id_to_bytes(t) for t in generated[start + 1 :]).decode("utf-8", "ignore")
        m = _OPEN.search(body)
        if not m or "\\" in m.group(2):
            return None
        return m.group(1), m.group(2), body

    @staticmethod
    def next_after_close(field: str, body: str) -> str:
        """JSON character that must follow the closing quote (keys are canonical-sorted:
        date, in_minutes, repeat, task, time; amount, inclusive, rate)."""
        if field in ("date", "amount"):
            return ","
        if field == "task" and '"scheduler.create"' in body and '"in_minutes"' not in body:
            return ","  # "time" always follows "task" unless the reminder is relative
        return "}"

    def continuations(self, field: str, partial: str) -> list[str]:
        """What may follow `partial` in each source (the value must start at a word start)."""
        if field in _CHOICE_FIELDS:
            return [d[len(partial):] for d in self.choices[field] if d.startswith(partial) and d != partial]
        out = []
        word = (lambda ch: ch.isalnum() or ch in "_-./\\") if field == "path" else str.isalnum
        for src in self.sources[field]:
            whole = field == "path" and src in self.listed  # a listed file name is copied whole
            if not partial:
                starts = [i for i, ch in enumerate(src) if ch.isalnum() and (i == 0 or not word(src[i - 1]))]
            else:
                starts = []
                i = src.find(partial)
                while i != -1:
                    if i == 0 or not word(src[i - 1]):
                        starts.append(i)
                    i = src.find(partial, i + 1)
            if whole:
                starts = [i for i in starts if i == 0]
            if field in ("text", "query") and (lab := label_end(src)) is not None:
                starts = [i for i in starts if i == lab]  # a labelled note is the whole content after the label
            if field == "task" and not partial:
                starts = [i for i in starts if not src[i].isdigit()]  # tasks start with a word, not "40 PM"
            for i in starts:
                rest = src[i + len(partial):]
                pats = {"task": (_STOP_ALL, _STOP_TASK), "text": (_STOP_ALL,), "query": (_STOP_ALL,),
                        "path": () if whole else (_STOP_PATH,)}[field]
                stops = [m.start() for p in pats if (m := p.search(src[i:]))]  # measured from the value's start
                if stops:
                    cut = min(stops) - len(partial)
                    if cut <= 0:
                        continue
                    rest = rest[:cut]
                if field == "path" and not self.complete(field, partial + rest):
                    continue  # only spans that are whole file names ("vet_visits.txt", not "the")
                out.append(rest)
        return out

    def complete(self, field: str, value: str) -> bool:
        if field in _CHOICE_FIELDS:
            return value in self.choices[field]
        if not value.strip() or value != value.strip():
            return False
        if field == "path":  # a whole file name: listed by a tool, or ending where the name ends in the message
            return value in self.listed or value in self.path_spans
        if field in ("text", "query"):  # notes and queries are copied verbatim to the end of the sentence
            return any(value == src[i:i + len(value)] and _ends_sentence(src[i + len(value):])
                       and label_end(src) in (None, i)
                       for src in self.sources[field] for i in self._starts(src, value))
        return copied_from(value, self.sources[field])

    @staticmethod
    def _starts(src: str, value: str) -> list[int]:
        out, i = [], src.find(value)
        while i != -1:
            if i == 0 or not src[i - 1].isalnum():
                out.append(i)
            i = src.find(value, i + 1)
        return out

    def __call__(self, generated: list[int], logits: torch.Tensor) -> torch.Tensor:
        state = self.open_field(generated)
        if state is None:
            return logits
        if state[0] in _CHOICE_FIELDS and not self.choices[state[0]] or state[0] not in _CHOICE_FIELDS and not self.sources[state[0]]:
            return logits  # nothing to choose from: leave the model free
        field, partial, body = state
        allowed: set[int] = set()
        for rest in self.continuations(field, partial):
            for n in range(1, min(len(rest), self.maxlen) + 1):
                allowed.update(self.pieces.get(rest[:n], ()))
        if self.complete(field, partial):
            need = self.next_after_close(field, body)
            closers = [t for t in self.closing if self.closer_text[t][1:2] == need]
            allowed.update(closers or [t for t in self.closing if self.closer_text[t] == '"'])
        allowed = {t for t in allowed if torch.isfinite(logits[t])}
        if not allowed:
            return logits  # never dead-end: fall back to the model's own choice
        idx = torch.tensor(sorted(allowed))
        out = torch.full_like(logits, float("-inf"))
        out[idx] = logits[idx]
        return out
