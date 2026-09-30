"""SwiGLU feed-forward: down( silu(gate(x)) * up(x) )."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from arouse.model.config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, f = cfg.d_model, cfg.ffn_size
        self.w_gate = nn.Linear(d, f, bias=cfg.bias)
        self.w_up = nn.Linear(d, f, bias=cfg.bias)
        self.w_down = nn.Linear(f, d, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
