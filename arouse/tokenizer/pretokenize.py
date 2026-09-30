"""Pre-tokenization: split text into chunks before BPE. Merges never cross chunks.

Stdlib `re` only. A "letter" is any Unicode letter or combining mark, so words in
Indic and other scripts with vowel signs (e.g. Hindi "हिन्दी") stay in one chunk.
The mark class is built from `unicodedata`; the final pattern string is saved
inside every tokenizer file, so a trained tokenizer never depends on the
Python version that loads it.
"""

from __future__ import annotations

import re
import sys
import unicodedata


def _mark_class() -> str:
    """Regex class body covering every Unicode combining mark (categories Mn, Mc, Me)."""
    ranges: list[tuple[int, int]] = []
    for cp in range(sys.maxunicode + 1):
        if unicodedata.category(chr(cp))[0] == "M":
            if ranges and ranges[-1][1] == cp - 1:
                ranges[-1] = (ranges[-1][0], cp)
            else:
                ranges.append((cp, cp))

    def esc(cp: int) -> str:
        return f"\\u{cp:04x}" if cp <= 0xFFFF else f"\\U{cp:08x}"

    return "".join(esc(a) if a == b else f"{esc(a)}-{esc(b)}" for a, b in ranges)


_M = _mark_class()
_LETTER = rf"(?:[^\W\d_]|[{_M}])"

PATTERN_V1 = (
    rf"[^\r\n\w{_M}]?{_LETTER}+"  # letters, optionally led by one space/punct char: ' hello', '"tool', "'s"
    rf"|_+{_LETTER}*"  # snake_case pieces: '_id'
    r"|\d"  # ONE digit per chunk: dates/times/numbers tokenize uniformly
    rf"| ?[^\s\w{_M}]+[\r\n]*"  # punctuation/symbol runs: '":"', '{"', ' ->'
    r"|\s*[\r\n]+"  # newlines (+ preceding spaces)
    r"|\s+(?!\S)"  # whitespace run, leaving one space to attach to the next word
    r"|\s+"
    r"|[\s\S]"  # safety net: never drop a character
)

_COMPILED: dict[str, re.Pattern[str]] = {}


def compile_pattern(pattern: str = PATTERN_V1) -> re.Pattern[str]:
    if pattern not in _COMPILED:
        _COMPILED[pattern] = re.compile(pattern)
    return _COMPILED[pattern]


def pretokenize(text: str, pattern: str = PATTERN_V1) -> list[str]:
    return compile_pattern(pattern).findall(text)
