"""Token embeddings and rotary positional encoding (RoPE).

There is no absolute position embedding: position enters only through RoPE,
which rotates query/key pairs by an angle proportional to position, so
attention scores depend on the *relative* distance between tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.weight)


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables; `forward(x, start_pos)` rotates x of shape (B, T, heads, head_dim).

    Half-split convention: dims [0, d/2) pair with [d/2, d).
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
        angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float64), inv_freq)  # (T, d/2)
        self.register_buffer("cos", angles.cos().float(), persistent=False)
        self.register_buffer("sin", angles.sin().float(), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        T = x.shape[1]
        cos = self.cos[start_pos : start_pos + T][None, :, None, :]
        sin = self.sin[start_pos : start_pos + T][None, :, None, :]
        xf = x.float()
        half = xf.shape[-1] // 2
        x1, x2 = xf[..., :half], xf[..., half:]
        out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.type_as(x)
