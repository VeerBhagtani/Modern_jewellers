"""Model architecture configuration.

Decoder-only Transformer: token embedding -> N x [RMSNorm -> causal self-attention
(RoPE, optional GQA) -> residual -> RMSNorm -> SwiGLU MLP -> residual] -> RMSNorm -> LM head.

Every architectural number lives here so scaling = editing YAML, not code.
`None` fields are "auto" and derived from other fields (see properties);
`resolved()` makes them explicit, which is what checkpoints must store.
"""

from __future__ import annotations

import dataclasses

from arouse.config import ConfigBase, ConfigError

# 64 special-token slots + 256 byte tokens: the smallest legal vocabulary.
MIN_VOCAB_SIZE = 64 + 256


@dataclasses.dataclass
class ModelConfig(ConfigBase):
    name: str = "arouse-v0.1"
    vocab_size: int = 32768
    context_length: int = 2048
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int | None = None  # None -> n_heads (MHA); fewer -> grouped-query attention
    ffn_hidden_size: int | None = None  # None -> ~8/3 * d_model rounded up to ffn_multiple_of
    ffn_multiple_of: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    bias: bool = False  # biases in attention/MLP linears
    tie_embeddings: bool = True  # LM head shares the embedding matrix
    init_std: float = 0.02

    # --- derived -------------------------------------------------------

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    @property
    def ffn_size(self) -> int:
        if self.ffn_hidden_size is not None:
            return self.ffn_hidden_size
        # SwiGLU has 3 matrices; 8/3*d keeps params equal to a 4*d GELU MLP.
        raw = int(8 * self.d_model / 3)
        m = self.ffn_multiple_of
        return ((raw + m - 1) // m) * m

    def resolved(self) -> ModelConfig:
        """Copy with every auto field made explicit."""
        return self.replace(n_kv_heads=self.kv_heads, ffn_hidden_size=self.ffn_size)

    # --- validation ----------------------------------------------------

    def validate(self) -> None:
        positive = ("vocab_size", "context_length", "d_model", "n_layers", "n_heads", "ffn_multiple_of")
        for key in positive:
            if getattr(self, key) <= 0:
                raise ConfigError(f"{key} must be > 0")
        if self.vocab_size < MIN_VOCAB_SIZE:
            raise ConfigError(f"vocab_size must be >= {MIN_VOCAB_SIZE} (special slots + bytes)")
        if self.d_model % self.n_heads:
            raise ConfigError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.head_dim % 2:
            raise ConfigError(f"head_dim ({self.head_dim}) must be even for RoPE")
        if self.n_kv_heads is not None and (
            self.n_kv_heads <= 0 or self.n_heads % self.n_kv_heads
        ):
            raise ConfigError(f"n_heads ({self.n_heads}) must be a multiple of n_kv_heads ({self.n_kv_heads})")
        if self.ffn_hidden_size is not None and self.ffn_hidden_size <= 0:
            raise ConfigError("ffn_hidden_size must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("dropout must be in [0, 1)")
        for key in ("rope_theta", "norm_eps", "init_std"):
            if getattr(self, key) <= 0:
                raise ConfigError(f"{key} must be > 0")

    # --- parameter accounting -----------------------------------------

    def parameter_counts(self) -> dict[str, int]:
        """Exact parameter count of the architecture described above."""
        d, f, b = self.d_model, self.ffn_size, int(self.bias)
        kv = self.kv_heads * self.head_dim
        attn = d * d + 2 * d * kv + d * d + b * (d + 2 * kv + d)  # q, k, v, o
        mlp = 3 * d * f + b * (2 * f + d)  # gate, up, down
        norms = 2 * d  # RMSNorm weights (pre-attn, pre-mlp)
        per_layer = attn + mlp + norms
        embedding = self.vocab_size * d
        lm_head = 0 if self.tie_embeddings else self.vocab_size * d
        layers = self.n_layers * per_layer
        total = embedding + layers + d + lm_head  # + final norm
        return {
            "embedding": embedding,
            "per_layer": per_layer,
            "layers": layers,
            "final_norm": d,
            "lm_head": lm_head,
            "non_embedding": total - embedding - lm_head,
            "total": total,
        }

    def num_parameters(self) -> int:
        return self.parameter_counts()["total"]

    def summary(self) -> str:
        c = self.parameter_counts()
        return (
            f"{self.name}: {c['total'] / 1e6:.1f}M params "
            f"({c['non_embedding'] / 1e6:.1f}M non-embedding) | "
            f"L={self.n_layers} d={self.d_model} heads={self.n_heads}/{self.kv_heads}kv "
            f"head_dim={self.head_dim} ffn={self.ffn_size} vocab={self.vocab_size} "
            f"ctx={self.context_length} tied={self.tie_embeddings}"
        )


PRESETS: dict[str, ModelConfig] = {
    # CPU-sized model for tests and pipeline debugging. Matches the tiny tokenizer.
    "arouse-tiny": ModelConfig(
        name="arouse-tiny",
        vocab_size=2048,
        context_length=256,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        ffn_multiple_of=32,
    ),
    # First real target: ~110M params.
    "arouse-v0.1": ModelConfig(),
}


def get_preset(name: str) -> ModelConfig:
    if name not in PRESETS:
        raise ConfigError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name]
