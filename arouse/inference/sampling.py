"""Next-token selection: greedy, temperature, top-k, top-p (nucleus)."""

from __future__ import annotations

import dataclasses

import torch

from arouse.config import ConfigBase, ConfigError


@dataclasses.dataclass
class SamplingParams(ConfigBase):
    max_new_tokens: int = 256
    temperature: float = 0.8  # 0 = greedy (deterministic)
    top_k: int = 0  # 0 = disabled
    top_p: float = 1.0  # 1 = disabled
    seed: int | None = None

    def validate(self) -> None:
        if self.max_new_tokens < 1:
            raise ConfigError("max_new_tokens must be >= 1")
        if self.temperature < 0:
            raise ConfigError("temperature must be >= 0")
        if self.top_k < 0:
            raise ConfigError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ConfigError("top_p must be in (0, 1]")


def filter_logits(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Return a copy of 1-D `logits` with tokens outside top-k / nucleus set to -inf."""
    logits = logits.clone()
    if top_k and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits[logits < kth] = float("-inf")
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        # drop a token if the tokens *before* it already cover top_p (always keeps the best one)
        drop = (torch.cumsum(probs, dim=-1) - probs) >= top_p
        logits[order[drop]] = float("-inf")
    return logits


def sample_next(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator | None = None) -> int:
    """Pick one token id from 1-D logits."""
    logits = logits.float()
    if params.temperature == 0:
        return int(torch.argmax(logits))
    logits = filter_logits(logits / params.temperature, params.top_k, params.top_p)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))
