"""Test doubles: a scripted language model that emits pre-planned token sequences."""

from __future__ import annotations

import torch

from arouse.agent.episode import turn_event
from arouse.inference import InferenceEngine
from arouse.model.config import get_preset
from arouse.model.transformer import ModelOutput
from arouse.protocol import Turn, encode_turn_body


class ScriptedTurnsModel(torch.nn.Module):
    """Each generation call (= each new KV cache) plays the next scripted token list."""

    def __init__(self, scripts: list[list[int]], context_length: int = 4096) -> None:
        super().__init__()
        self.config = get_preset("arouse-tiny").replace(context_length=context_length)
        self.scripts = scripts
        self.call = -1
        self.step = 0
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.prompts: list[int] = []  # prompt length of each call

    def new_kv_cache(self):  # noqa: ANN201
        self.call += 1
        self.step = 0
        return None

    def forward(self, ids, targets=None, *, start_pos=0, kv_cache=None):  # noqa: ANN001, ANN201
        if start_pos == 0:
            self.prompts.append(ids.shape[1])
        logits = torch.zeros(1, ids.shape[1], self.config.vocab_size)
        script = self.scripts[min(self.call, len(self.scripts) - 1)]
        if self.step < len(script):
            logits[0, -1, script[self.step]] = 10.0
        self.step += 1
        return ModelOutput(logits, None)


def scripted_engine(tok, turns_or_ids) -> InferenceEngine:  # noqa: ANN001
    """Engine whose successive generations produce the given Turns (or raw token lists)."""
    scripts = [encode_turn_body(tok, t) if isinstance(t, Turn) else list(t) for t in turns_or_ids]
    return InferenceEngine(ScriptedTurnsModel(scripts), tok)


def as_events(*turns: Turn) -> list[dict]:
    return [turn_event(t) for t in turns]
