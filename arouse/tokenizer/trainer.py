"""Tokenizer training: read corpus -> count pre-token chunks -> learn BPE merges.

Corpus files:
  *.txt    whole file is one document
  *.jsonl  one JSON object per line; its "text" field is a document
Special-token strings in the corpus are treated as hard boundaries and never learned.
"""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from arouse.config import ConfigBase, ConfigError
from arouse.tokenizer.bpe import train_bpe
from arouse.tokenizer.pretokenize import PATTERN_V1, compile_pattern
from arouse.tokenizer.special_tokens import split_on_special
from arouse.tokenizer.tokenizer import BYTE_OFFSET, FIRST_MERGE_ID, ArouseTokenizer


@dataclasses.dataclass
class TokenizerTrainingConfig(ConfigBase):
    vocab_size: int = 32768
    corpus: list[str] = dataclasses.field(default_factory=list)  # files or globs (relative to cwd)
    output_dir: str = "artifacts/tokenizers/arouse"
    min_frequency: int = 2
    max_bytes: int | None = None  # stop reading corpus after this many bytes

    def validate(self) -> None:
        if self.vocab_size < FIRST_MERGE_ID:
            raise ConfigError(f"vocab_size must be >= {FIRST_MERGE_ID} (64 special + 256 bytes)")
        if self.min_frequency < 1:
            raise ConfigError("min_frequency must be >= 1")
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise ConfigError("max_bytes must be > 0")


def resolve_corpus(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat, recursive=True))
        if not matches:
            raise ConfigError(f"corpus pattern matched no files: {pat}")
        files.extend(Path(m) for m in matches if Path(m).is_file())
    return files


def iter_documents(path: Path) -> Iterator[str]:
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if line.strip():
                    obj = json.loads(line)
                    if not isinstance(obj.get("text"), str):
                        raise ConfigError(f"{path}:{n}: missing string field 'text'")
                    yield obj["text"]
    else:
        yield path.read_text(encoding="utf-8")


def count_chunks(documents: Iterator[str], pattern: str = PATTERN_V1) -> Counter[bytes]:
    regex = compile_pattern(pattern)
    counts: Counter[bytes] = Counter()
    for doc in documents:
        for piece, is_special in split_on_special(doc):
            if not is_special:
                counts.update(c.encode("utf-8") for c in regex.findall(piece))
    return counts


def train_tokenizer(cfg: TokenizerTrainingConfig) -> ArouseTokenizer:
    files = resolve_corpus(cfg.corpus)
    if not files:
        raise ConfigError("empty corpus")
    sources = []
    total = 0

    def docs() -> Iterator[str]:
        nonlocal total
        for f in files:
            raw = f.read_bytes()
            sources.append({"path": f.as_posix(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
            for doc in iter_documents(f):
                if cfg.max_bytes is not None and total >= cfg.max_bytes:
                    return
                total += len(doc.encode("utf-8"))
                yield doc

    counts = count_chunks(docs())
    merges = train_bpe(counts, cfg.vocab_size - FIRST_MERGE_ID, byte_offset=BYTE_OFFSET, min_frequency=cfg.min_frequency)
    info = {
        "target_vocab_size": cfg.vocab_size,
        "min_frequency": cfg.min_frequency,
        "bytes_used": total,
        "unique_chunks": len(counts),
        "sources": sources,
    }
    return ArouseTokenizer(merges, training_info=info)
