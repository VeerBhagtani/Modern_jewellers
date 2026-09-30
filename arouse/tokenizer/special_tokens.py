"""Arouse special tokens.

IDs 0..63 are permanently reserved for special tokens, so byte and merge IDs never
shift when a special token is added (a reserved slot is renamed instead).

Sequence grammar (see docs/tokenizer.md):

    <|bos|> segment* <|eos|>
    segment := ROLE content <|end|>
    ROLE    := system | tools | context | memory | state | user | tool_result | tool_error | arouse

    arouse turn := <|arouse|> [<|plan|> text] [<|verify|> text] (ACTION json | text) <|end|>
    ACTION      := <|tool_call|> | <|ask_user|> | <|finish|> | <|fail|>

Sub-segments inside an arouse turn need no closing tag: each runs until the next
special token. The action type is a single token, so the model's most important
decision (call a tool / ask / finish / fail) is one prediction that decoding can
constrain and evaluation can score directly.
"""

from __future__ import annotations

import re
from enum import IntEnum

NUM_SPECIAL_SLOTS = 64
SPECIAL_TOKENS_VERSION = 1


class Special(IntEnum):
    # control
    PAD = 0  # batch padding; never predicted
    BOS = 1  # start of sequence
    EOS = 2  # end of document / episode
    END = 3  # closes any segment / turn
    # input segments (written by user or runtime, never generated)
    SYSTEM = 4  # operator instructions
    TOOLS = 5  # JSON list of available tool schemas
    CONTEXT = 6  # runtime facts: current datetime, timezone, locale
    MEMORY = 7  # retrieved long-term memory
    STATE = 8  # current task-state JSON (resume without restarting)
    USER = 9  # user message
    TOOL_RESULT = 10  # successful tool output (JSON)
    TOOL_ERROR = 11  # failed tool output (JSON); distinct token = hard failure signal
    # model turn
    AROUSE = 12  # start of Arouse's turn
    PLAN = 13  # short plan / reasoning
    VERIFY = 14  # check of observations before claiming completion
    # actions: followed by a JSON body, then <|end|>
    TOOL_CALL = 15  # {"tool": ..., "arguments": {...}}
    ASK_USER = 16  # {"question": ...}
    FINISH = 17  # {"result": ...}  task verified complete
    FAIL = 18  # {"error": ...}   task cannot be completed

    @property
    def text(self) -> str:
        return f"<|{self.name.lower()}|>"


INPUT_SEGMENTS = (
    Special.SYSTEM,
    Special.TOOLS,
    Special.CONTEXT,
    Special.MEMORY,
    Special.STATE,
    Special.USER,
    Special.TOOL_RESULT,
    Special.TOOL_ERROR,
)
REASONING = (Special.PLAN, Special.VERIFY)
ACTIONS = (Special.TOOL_CALL, Special.ASK_USER, Special.FINISH, Special.FAIL)

assert len(Special) <= NUM_SPECIAL_SLOTS


def special_token_texts() -> list[str]:
    """Text for all 64 slots, indexed by ID. Unused slots are `<|reserved_N|>`."""
    named = {t.value: t.text for t in Special}
    return [named.get(i, f"<|reserved_{i}|>") for i in range(NUM_SPECIAL_SLOTS)]


_SPLIT_RE = re.compile(
    "(" + "|".join(re.escape(t) for t in sorted(special_token_texts(), key=len, reverse=True)) + ")"
)


def split_on_special(text: str) -> list[tuple[str, bool]]:
    """Split text into (piece, is_special) pairs, dropping empty pieces."""
    return [(p, i % 2 == 1) for i, p in enumerate(_SPLIT_RE.split(text)) if p]
