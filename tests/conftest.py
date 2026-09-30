from __future__ import annotations

from pathlib import Path

import pytest

from arouse.tokenizer import ArouseTokenizer, TokenizerTrainingConfig, train_tokenizer

ROOT = Path(__file__).resolve().parents[1]
TINY_DATA = ROOT / "datasets" / "tokenizer_tiny"


@pytest.fixture(scope="session")
def tiny_tokenizer() -> ArouseTokenizer:
    cfg = TokenizerTrainingConfig(
        vocab_size=2048,
        corpus=[str(TINY_DATA / "*.txt"), str(TINY_DATA / "*.jsonl")],
        output_dir="unused",
    )
    return train_tokenizer(cfg)
