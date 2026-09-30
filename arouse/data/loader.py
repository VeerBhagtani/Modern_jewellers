"""Mixture batch sampling over packed per-source streams.

Batch contents are a pure function of (seed, step): resuming at step k yields exactly
the batches an uninterrupted run would have seen. Nothing to checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from arouse.config import ConfigError
from arouse.data.prepare import FORMAT, FORMAT_VERSION
from arouse.model.loss import IGNORE_INDEX


@dataclass
class Batch:
    inputs: torch.Tensor  # (B, T) int64
    targets: torch.Tensor  # (B, T) int64, IGNORE_INDEX where not trained

    @property
    def target_tokens(self) -> int:
        return int((self.targets != IGNORE_INDEX).sum())


class PackedStream:
    """One source/split: a memory-mapped token stream + mask."""

    def __init__(self, directory: Path, name: str, split: str, dtype: str, window: str = "random") -> None:
        self.window_mode = window
        docs_path = directory / f"{name}.{split}.docs.bin"
        self.doc_starts = np.fromfile(docs_path, dtype=np.uint64) if docs_path.exists() else np.zeros(0, np.uint64)
        tok_path = directory / f"{name}.{split}.tokens.bin"
        self.tokens = np.memmap(tok_path, dtype=np.dtype(dtype), mode="r") if tok_path.stat().st_size else np.zeros(0, np.uint16)
        mask_path = directory / f"{name}.{split}.mask.bin"
        self.mask = np.memmap(mask_path, dtype=np.uint8, mode="r") if mask_path.stat().st_size else np.zeros(0, np.uint8)
        if len(self.tokens) != len(self.mask):
            raise ConfigError(f"{name}.{split}: tokens/mask length mismatch")
        self.name = name

    def __len__(self) -> int:
        return len(self.tokens)

    def random_start(self, g: torch.Generator) -> int:
        if self.window_mode == "doc_start" and len(self.doc_starts):
            return int(self.doc_starts[int(torch.randint(0, len(self.doc_starts), (1,), generator=g))])
        return int(torch.randint(0, len(self.tokens), (1,), generator=g))

    def window(self, start: int, length: int) -> tuple[np.ndarray, np.ndarray]:
        """`length` tokens from `start`, wrapping around (short sources are tiled)."""
        idx = (np.arange(start, start + length) % len(self.tokens))
        return np.asarray(self.tokens[idx], dtype=np.int64), np.asarray(self.mask[idx], dtype=np.int64)


def _to_batch(windows: list[tuple[np.ndarray, np.ndarray]]) -> Batch:
    toks = torch.from_numpy(np.stack([w[0] for w in windows]))
    mask = torch.from_numpy(np.stack([w[1] for w in windows]))
    targets = toks[:, 1:].clone()
    targets[mask[:, 1:] == 0] = IGNORE_INDEX
    return Batch(toks[:, :-1].contiguous(), targets)


class MixtureLoader:
    def __init__(self, data_dir: str | Path, seq_len: int, seed: int = 0) -> None:
        self.dir = Path(data_dir)
        try:
            self.meta: dict[str, Any] = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        except FileNotFoundError as e:
            raise ConfigError(f"no prepared data at {self.dir} (run `arouse data prepare`)") from e
        if self.meta.get("format") != FORMAT or self.meta.get("format_version") != FORMAT_VERSION:
            raise ConfigError("unsupported prepared-data format")
        self.seq_len = seq_len
        self.seed = seed
        dtype = self.meta["token_dtype"]
        self.train: dict[str, PackedStream] = {}
        self.val: dict[str, PackedStream] = {}
        weights = []
        for s in self.meta["sources"]:
            tr = PackedStream(self.dir, s["name"], "train", dtype, s.get("window", "random"))
            if len(tr) >= 2:
                self.train[s["name"]] = tr
                weights.append(s["weight"])
            va = PackedStream(self.dir, s["name"], "val", dtype)
            if len(va) >= 2:
                self.val[s["name"]] = va
        if not self.train:
            raise ConfigError("prepared data has no training tokens")
        self.names = list(self.train)
        w = torch.tensor(weights, dtype=torch.float64)
        self.probs = w / w.sum()

    @property
    def data_fingerprint(self) -> str:
        return self.meta["data_fingerprint"]

    @property
    def tokenizer_fingerprint(self) -> str:
        return self.meta["tokenizer_fingerprint"]

    def train_batch(self, step: int, batch_size: int, micro: int = 0) -> Batch:
        digest = hashlib.sha256(f"{self.seed}:{step}:{micro}".encode()).digest()
        g = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF)
        picks = torch.multinomial(self.probs, batch_size, replacement=True, generator=g).tolist()
        windows = []
        for p in picks:
            stream = self.train[self.names[p]]
            windows.append(stream.window(stream.random_start(g), self.seq_len + 1))
        return _to_batch(windows)

    def val_batches(self, batch_size: int, max_batches: int) -> dict[str, list[Batch]]:
        """Fixed, non-overlapping windows per source (identical every evaluation)."""
        out: dict[str, list[Batch]] = {}
        n = self.seq_len + 1
        for name, stream in self.val.items():
            starts = list(range(0, max(len(stream) - n + 1, 1), n))[: batch_size * max_batches]
            windows = [stream.window(s, n) for s in starts]
            out[name] = [_to_batch(windows[i : i + batch_size]) for i in range(0, len(windows), batch_size)]
        return out
