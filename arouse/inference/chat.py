"""Chat prompt construction.

    <|bos|> [<|system|>..<|end|>] (<|user|>..<|end|> | <|arouse|>..<|end|>)* <|arouse|>

User and system text is encoded with allow_special=False: a user typing "<|finish|>"
cannot inject control tokens. Assistant history may contain the model's own
structural tokens (<|plan|>, <|tool_call|>, ...), so it is encoded with them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from arouse.tokenizer import ArouseTokenizer, Special

ROLES = {"system": Special.SYSTEM, "user": Special.USER, "assistant": Special.AROUSE}


class PromptTooLong(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {sorted(ROLES)}, got {self.role!r}")
        if not isinstance(self.content, str):
            raise ValueError("content must be a string")


def _encode_message(tok: ArouseTokenizer, m: Message) -> list[int]:
    body = tok.encode(m.content, allow_special=(m.role == "assistant"))
    return [ROLES[m.role], *body, Special.END]


def encode_chat(tok: ArouseTokenizer, messages: Sequence[Message], *, max_tokens: int | None = None) -> list[int]:
    """Encode a conversation and open Arouse's turn.

    If `max_tokens` is given and the prompt is too long, the oldest non-system
    messages are dropped (the system prompt and the latest message are always kept).
    """
    if not messages:
        raise ValueError("at least one message is required")
    encoded = [(m, _encode_message(tok, m)) for m in messages]

    def total(items: list[tuple[Message, list[int]]]) -> int:
        return 2 + sum(len(ids) for _, ids in items)  # + BOS + opening <|arouse|>

    if max_tokens is not None:
        while total(encoded) > max_tokens:
            droppable = [i for i, (m, _) in enumerate(encoded[:-1]) if m.role != "system"]
            if not droppable:
                raise PromptTooLong(f"prompt needs {total(encoded)} tokens, limit is {max_tokens}")
            del encoded[droppable[0]]

    ids: list[int] = [Special.BOS]
    for _, m_ids in encoded:
        ids.extend(m_ids)
    ids.append(Special.AROUSE)
    return ids
