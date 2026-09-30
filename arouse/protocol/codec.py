"""Token-level codec between protocol objects and model token sequences.

Encoding is piecewise: structure comes from special-token IDs, while every piece of
text (plan, JSON bodies, user messages, tool output) is encoded with
allow_special=False. A string such as "<|finish|>" inside a user message or a tool
argument can therefore never become a control token.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from arouse.protocol.actions import Action, ProtocolError, Turn, canonical_json
from arouse.tokenizer import ArouseTokenizer, Special

ACTION_TOKENS = {
    "tool_call": Special.TOOL_CALL,
    "ask_user": Special.ASK_USER,
    "finish": Special.FINISH,
    "fail": Special.FAIL,
}
_TOKEN_TO_ACTION = {int(v): k for k, v in ACTION_TOKENS.items()}


def encode_turn_body(tok: ArouseTokenizer, turn: Turn) -> list[int]:
    """Tokens of a turn after <|arouse|>, including the closing <|end|>."""
    ids: list[int] = []
    if turn.plan:
        ids += [Special.PLAN, *tok.encode(turn.plan)]
    if turn.verify:
        ids += [Special.VERIFY, *tok.encode(turn.verify)]
    ids += [ACTION_TOKENS[turn.action.type], *tok.encode(canonical_json(turn.action.body())), Special.END]
    return ids


def render_turn_text(turn: Turn) -> str:
    """Text form of a turn (for logs / data files). Tokenizing it with allow_special=True
    equals encode_turn_body() whenever no text field contains special-token strings."""
    out = Special.AROUSE.text
    if turn.plan:
        out += Special.PLAN.text + turn.plan
    if turn.verify:
        out += Special.VERIFY.text + turn.verify
    return out + ACTION_TOKENS[turn.action.type].text + canonical_json(turn.action.body()) + Special.END.text


def decode_turn(tok: ArouseTokenizer, ids: Sequence[int]) -> Turn:
    """Parse generated tokens (after <|arouse|>, END optional) into a validated Turn."""
    ids = list(ids)
    if ids and ids[-1] == Special.END:
        ids = ids[:-1]
    sections: list[tuple[int, list[int]]] = []
    for t in ids:
        if tok.is_special(t):
            sections.append((t, []))
        elif not sections:
            raise ProtocolError("turn must start with <|plan|>, <|verify|> or an action token")
        else:
            sections[-1][1].append(t)

    plan = verify = None
    action: Action | None = None
    for i, (marker, body) in enumerate(sections):
        text = tok.decode(body)
        if marker == Special.PLAN and plan is None and action is None:
            plan = text.strip()
        elif marker == Special.VERIFY and verify is None and action is None:
            verify = text.strip()
        elif marker in _TOKEN_TO_ACTION and action is None and i == len(sections) - 1:
            try:
                fields = json.loads(text)
            except json.JSONDecodeError as e:
                raise ProtocolError(f"action body is not valid JSON: {e}") from e
            if not isinstance(fields, dict) or "type" in fields:
                raise ProtocolError("action body must be an object without 'type'")
            action = Action.from_dict({"type": _TOKEN_TO_ACTION[marker], **fields})
        else:
            raise ProtocolError(f"unexpected token {tok.id_to_bytes(marker).decode()} in turn")
    if action is None:
        raise ProtocolError("turn has no action")
    return Turn(action=action, plan=plan or None, verify=verify or None)


def action_from_json(payload: Any) -> Action:
    return Action.from_dict(payload)
