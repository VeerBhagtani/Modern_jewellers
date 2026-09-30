"""LM head: hidden state -> vocabulary logits, optionally tied to the embedding matrix."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, tied_weight: nn.Parameter | None = None) -> None:
        super().__init__()
        # Tied: the very same Parameter object as the embedding (counted/updated once).
        self.weight = tied_weight if tied_weight is not None else nn.Parameter(torch.empty(vocab_size, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)
