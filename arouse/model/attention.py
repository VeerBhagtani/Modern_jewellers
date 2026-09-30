"""Causal multi-head self-attention with RoPE, grouped-query attention and a KV cache.

Two numerically equivalent implementations:
  "manual" - explicit softmax(QK^T / sqrt(d) + causal_mask) V, easy to read and debug
  "sdpa"   - torch's fused scaled_dot_product_attention (fast path, default)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from arouse.model.config import ModelConfig
from arouse.model.embeddings import RotaryEmbedding

ATTN_IMPLS = ("sdpa", "manual")


def causal_mask(q_len: int, k_len: int, start_pos: int, device: torch.device) -> torch.Tensor:
    """Bool (q_len, k_len); True = may attend. Query i sits at absolute position start_pos + i."""
    q_pos = torch.arange(start_pos, start_pos + q_len, device=device)[:, None]
    k_pos = torch.arange(k_len, device=device)[None, :]
    return k_pos <= q_pos


class LayerKVCache:
    """Keys/values of all previous positions for one layer, shape (B, S, kv_heads, head_dim)."""

    def __init__(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.k = k if self.k is None else torch.cat((self.k, k), dim=1)
        self.v = v if self.v is None else torch.cat((self.v, v), dim=1)
        return self.k, self.v

    def __len__(self) -> int:
        return 0 if self.k is None else self.k.shape[1]


class KVCache:
    def __init__(self, n_layers: int) -> None:
        self.layers = [LayerKVCache() for _ in range(n_layers)]

    def __len__(self) -> int:
        return len(self.layers[0])


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, impl: str = "sdpa") -> None:
        super().__init__()
        if impl not in ATTN_IMPLS:
            raise ValueError(f"attention impl must be one of {ATTN_IMPLS}")
        self.impl = impl
        self.n_heads, self.kv_heads, self.head_dim = cfg.n_heads, cfg.kv_heads, cfg.head_dim
        d, hd = cfg.d_model, cfg.head_dim
        self.wq = nn.Linear(d, self.n_heads * hd, bias=cfg.bias)
        self.wk = nn.Linear(d, self.kv_heads * hd, bias=cfg.bias)
        self.wv = nn.Linear(d, self.kv_heads * hd, bias=cfg.bias)
        self.wo = nn.Linear(self.n_heads * hd, d, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        start_pos: int = 0,
        cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = rope(self.wq(x).view(B, T, self.n_heads, self.head_dim), start_pos)
        k = rope(self.wk(x).view(B, T, self.kv_heads, self.head_dim), start_pos)
        v = self.wv(x).view(B, T, self.kv_heads, self.head_dim)
        if cache is not None:
            k, v = cache.append(k, v)

        # (B, heads, len, head_dim); GQA: each kv head serves n_heads / kv_heads query heads
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.kv_heads != self.n_heads:
            rep = self.n_heads // self.kv_heads
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)

        p = self.dropout if self.training else 0.0
        S = k.shape[2]
        if self.impl == "sdpa":
            if start_pos == 0 and T == S:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
            else:
                mask = causal_mask(T, S, start_pos, x.device)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~causal_mask(T, S, start_pos, x.device), float("-inf"))
            probs = F.dropout(torch.softmax(scores.float(), dim=-1).type_as(q), p)
            y = probs @ v

        return self.wo(y.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim))
