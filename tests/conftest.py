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


@pytest.fixture(scope="session")
def tiny_data_dir(tiny_tokenizer, tmp_path_factory):
    """Prepared (packed) data from two tiny sources, for training tests."""
    from arouse.data.config import DataConfig
    from arouse.data.prepare import prepare

    root = tmp_path_factory.mktemp("data")
    tiny_tokenizer.save(root / "tok")
    cfg = DataConfig.from_dict({
        "name": "test",
        "tokenizer": str(root / "tok"),
        "output_dir": str(root / "packed"),
        "val_fraction": 0.1,
        "sources": [
            {"name": "agent", "paths": [str(TINY_DATA / "agent_trajectories.jsonl")], "format": "jsonl",
             "category": "agent", "origin": "generated", "license": "project-owned", "allow_special": True, "weight": 3.0},
            {"name": "general", "paths": [str(TINY_DATA / "general.txt")], "split": "paragraphs",
             "category": "general", "origin": "hand-written", "license": "project-owned"},
        ],
    })
    prepare(cfg)
    return root / "packed"
