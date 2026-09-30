"""ArouseTransformer: the full decoder-only language model.

    ids -> TokenEmbedding -> dropout -> N x TransformerBlock -> RMSNorm -> LMHead -> logits
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn

from arouse.model.attention import KVCache
from arouse.model.config import ModelConfig
from arouse.model.embeddings import RotaryEmbedding, TokenEmbedding
from arouse.model.lm_head import LMHead
from arouse.model.loss import lm_loss
from arouse.model.norm import RMSNorm
from arouse.model.transformer_block import TransformerBlock


class ModelOutput(NamedTuple):
    logits: torch.Tensor  # (B, T, vocab)
    loss: torch.Tensor | None


class ArouseTransformer(nn.Module):
    def __init__(self, config: ModelConfig, attn_impl: str = "sdpa") -> None:
        super().__init__()
        cfg = config.resolved()
        self.config = cfg
        self.embed = TokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.embed_drop = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.context_length, cfg.rope_theta)
        self.layers = nn.ModuleList(TransformerBlock(cfg, attn_impl) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = LMHead(cfg.d_model, cfg.vocab_size, self.embed.weight if cfg.tie_embeddings else None)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """N(0, init_std) for matrices; residual output projections scaled by 1/sqrt(2*n_layers)
        so the residual stream's variance does not grow with depth. Biases 0, norm gains 1."""
        std = self.config.init_std
        for name, p in self.named_parameters():
            if name.endswith("norm.weight"):
                nn.init.ones_(p)
            elif p.dim() == 1:
                nn.init.zeros_(p)
            elif name.endswith(("attn.wo.weight", "mlp.w_down.weight")):
                nn.init.normal_(p, 0.0, std / math.sqrt(2 * self.config.n_layers))
            else:
                nn.init.normal_(p, 0.0, std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        start_pos: int = 0,
        kv_cache: KVCache | None = None,
    ) -> ModelOutput:
        """input_ids: (B, T) int64. targets: (B, T) aligned next-token ids or IGNORE_INDEX.
        start_pos/kv_cache: incremental decoding (positions start_pos .. start_pos+T-1)."""
        B, T = input_ids.shape
        if start_pos + T > self.config.context_length:
            raise ValueError(f"sequence end {start_pos + T} exceeds context_length {self.config.context_length}")
        if kv_cache is not None and len(kv_cache) != start_pos:
            raise ValueError(f"kv_cache holds {len(kv_cache)} positions but start_pos={start_pos}")

        x = self.embed_drop(self.embed(input_ids))
        for i, layer in enumerate(self.layers):
            x = layer(x, self.rope, start_pos, kv_cache.layers[i] if kv_cache is not None else None)
        logits = self.lm_head(self.norm(x))
        loss = lm_loss(logits, targets) if targets is not None else None
        return ModelOutput(logits, loss)

    def new_kv_cache(self) -> KVCache:
        return KVCache(self.config.n_layers)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())  # shared (tied) params counted once
