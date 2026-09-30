"""Document cleaning, dedup keys and deterministic train/val assignment."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # C0 controls except \t \n \r
_WS = re.compile(r"\s+")


def clean_text(text: str, *, min_chars: int = 1, max_bad_char_ratio: float = 0.1) -> str | None:
    """Normalise a document; None = drop it.

    - Unicode NFC (composed form, so "é" is always the same bytes)
    - \\r\\n and \\r -> \\n
    - strip control characters and trailing whitespace on each line
    - drop documents that are too short or mostly garbage (control / U+FFFD chars)
    """
    if not text:
        return None
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    bad = len(_CONTROL.findall(text)) + text.count("�")
    if bad / len(text) > max_bad_char_ratio:
        return None
    text = _CONTROL.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")
    if len(text.strip()) < min_chars:
        return None
    return text


def dedup_key(text: str) -> str:
    """Exact-duplicate key, insensitive to case and whitespace differences."""
    return hashlib.sha1(_WS.sub(" ", text.lower()).strip().encode()).hexdigest()


def is_validation(text: str, val_fraction: float) -> bool:
    """Stable split from content alone: same document -> same split on every run and machine."""
    if val_fraction <= 0:
        return False
    h = int.from_bytes(hashlib.sha256(("split:" + dedup_key(text)).encode()).digest()[:8], "big")
    return (h % 1_000_000) < val_fraction * 1_000_000
