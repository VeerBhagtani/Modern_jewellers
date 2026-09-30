"""Pre-norm residual block: x + Attn(Norm(x)), then x + MLP(Norm(x))."""

from __future__ import annotations

import torch
from torch import nn

from arouse.model.attention import CausalSelfAttention, LayerKVCache
from arouse.model.config import ModelConfig
from arouse.model.embeddings import RotaryEmbedding
from arouse.model.mlp import SwiGLU
from arouse.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, attn_impl: str = "sdpa") -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg, attn_impl)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        start_pos: int = 0,
        cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        x = x + self.resid_drop(self.attn(self.attn_norm(x), rope, start_pos, cache))
        x = x + self.resid_drop(self.mlp(self.mlp_norm(x)))
        return x
