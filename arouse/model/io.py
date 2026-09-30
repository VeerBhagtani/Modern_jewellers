"""Model export format: everything needed to run a model, in one directory.

    config.yaml    resolved ModelConfig
    model.pt       state_dict (loaded with weights_only=True: no pickled code)
    tokenizer/     the exact tokenizer the model was trained with
    meta.json      format version, tokenizer fingerprint, training status

Training checkpoints (Milestone 3) add optimizer/scheduler/RNG state on top of this.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from arouse.model.config import ModelConfig
from arouse.model.transformer import ArouseTransformer
from arouse.tokenizer import ArouseTokenizer

FORMAT = "arouse-model"
FORMAT_VERSION = 1


class ModelLoadError(ValueError):
    pass


def check_compatible(config: ModelConfig, tokenizer: ArouseTokenizer) -> None:
    if config.vocab_size < tokenizer.vocab_size:
        raise ModelLoadError(
            f"model vocab_size {config.vocab_size} < tokenizer vocab_size {tokenizer.vocab_size}"
        )


def save_pretrained(
    directory: str | Path,
    model: ArouseTransformer,
    tokenizer: ArouseTokenizer,
    *,
    trained: bool,
    train_steps: int = 0,
    notes: str = "",
) -> Path:
    check_compatible(model.config, tokenizer)
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    model.config.to_yaml(d / "config.yaml")
    torch.save(model.state_dict(), d / "model.pt")
    tokenizer.save(d / "tokenizer")
    meta: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "name": model.config.name,
        "num_parameters": model.num_parameters(),
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "trained": trained,
        "train_steps": train_steps,
        "notes": notes,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return d


def load_pretrained(
    directory: str | Path, *, device: str | torch.device = "cpu", attn_impl: str = "sdpa"
) -> tuple[ArouseTransformer, ArouseTokenizer, dict[str, Any]]:
    """Load a model export, or the newest checkpoint of a training run directory."""
    d = Path(directory)
    if (d / "LATEST").exists():  # a training run directory: use its newest checkpoint
        d = d / (d / "LATEST").read_text(encoding="utf-8").strip()
    try:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ModelLoadError(f"not a model directory (missing meta.json): {d}") from e
    if meta.get("format") != FORMAT or meta.get("format_version") != FORMAT_VERSION:
        raise ModelLoadError(f"unsupported model format: {meta.get('format')} v{meta.get('format_version')}")

    config = ModelConfig.from_yaml(d / "config.yaml")
    tokenizer = ArouseTokenizer.load(d / "tokenizer")
    if tokenizer.fingerprint() != meta.get("tokenizer_fingerprint"):
        raise ModelLoadError("tokenizer fingerprint does not match the one the model was saved with")
    check_compatible(config, tokenizer)

    model = ArouseTransformer(config, attn_impl=attn_impl)
    state = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), tokenizer, meta
