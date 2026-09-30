"""Copy-constrained decoding for free-text arguments.

While the model is writing the value of a copy field ("task", "text", "path") inside a
tool call, the only tokens allowed are those that continue a phrase from the allowed
sources (the user's words in this request; for paths also file names returned by a
tool), plus the closing quote once the value is a complete phrase ending at a boundary.
The model still chooses which phrase to copy and where it ends; it cannot invent words.

"date" values are restricted to dates that exist in the conversation: today, tomorrow,
the upcoming weekdays from the runtime calendar, and the next occurrence of any explicit
date the user wrote ("October 5", "5th of October"). The model still chooses which one.
Everything else in the turn is decoded unconstrained.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

import torch

from arouse.agent.grounding import copied_from, current_request
from arouse.tokenizer import ArouseTokenizer, Special

_OPEN = re.compile(r'"(task|text|path|date)":"((?:[^"\\]|\\.)*)$')
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]
_MONTH_RE = "|".join(sorted({*_MONTHS, *(m[:3] for m in _MONTHS), "sept"}, key=len, reverse=True))
_DATE_PATTERNS = [
    re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.I),  # October 5
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_RE})\b", re.I),  # 5 October / 5th of October
]
_CLOSE = re.compile(r'^"([,}\]].*)?$', re.S)  # closes the string; the rest is ordinary JSON structure
_VOCAB: dict[str, tuple[dict[str, list[int]], list[int], int]] = {}


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
        req = current_request(events)
        said = [ev["content"] for ev in req if ev["type"] == "user"]
        listed = [f for ev in req if ev["type"] == "tool_result" for f in ev["content"].get("files", [])]
        self.sources = {"task": said, "text": said, "path": said + listed}
        self.listed = set(listed)
        dates = calendar_dates(context)
        if dates:
            today = datetime.strptime(context["now"][:10], "%Y-%m-%d").date()
            dates |= mentioned_dates(said, today)
        self.dates = sorted(dates)

    def open_field(self, generated: list[int]) -> tuple[str, str] | None:
        """(field, partial value) if the model is currently inside a copy-field string."""
        if Special.TOOL_CALL not in generated:
            return None
        start = len(generated) - 1 - generated[::-1].index(Special.TOOL_CALL)
        body = b"".join(self.tok.id_to_bytes(t) for t in generated[start + 1 :]).decode("utf-8", "ignore")
        m = _OPEN.search(body)
        if not m or "\\" in m.group(2):
            return None
        return m.group(1), m.group(2)

    def continuations(self, field: str, partial: str) -> list[str]:
        """What may follow `partial` in each source (the value must start at a word start)."""
        if field == "date":
            return [d[len(partial):] for d in self.dates if d.startswith(partial) and d != partial]
        out = []
        for src in self.sources[field]:
            if not partial:
                starts = [i for i, ch in enumerate(src) if ch.isalnum() and (i == 0 or not src[i - 1].isalnum())]
            else:
                starts = []
                i = src.find(partial)
                while i != -1:
                    if i == 0 or not src[i - 1].isalnum():
                        starts.append(i)
                    i = src.find(partial, i + 1)
            out += [src[i + len(partial):] for i in starts]
        return out

    def complete(self, field: str, value: str) -> bool:
        if field == "date":
            return value in self.dates
        if not value.strip() or value != value.strip():
            return False
        if field == "path":
            return value in self.listed or ("." in value and any(value in s for s in self.sources["path"]))
        return copied_from(value, self.sources[field])

    def __call__(self, generated: list[int], logits: torch.Tensor) -> torch.Tensor:
        state = self.open_field(generated)
        if state is None or (state[0] == "date" and not self.dates) or (state[0] != "date" and not self.sources[state[0]]):
            return logits
        field, partial = state
        allowed: set[int] = set()
        for rest in self.continuations(field, partial):
            for n in range(1, min(len(rest), self.maxlen) + 1):
                allowed.update(self.pieces.get(rest[:n], ()))
        if self.complete(field, partial):
            allowed.update(self.closing)
        allowed = {t for t in allowed if torch.isfinite(logits[t])}
        if not allowed:
            return logits  # never dead-end: fall back to the model's own choice
        idx = torch.tensor(sorted(allowed))
        out = torch.full_like(logits, float("-inf"))
        out[idx] = logits[idx]
        return out
