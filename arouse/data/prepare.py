"""Dataset preparation: sources -> clean -> dedup -> split -> tokenize -> packed binary streams.

Output directory:
    meta.json                     provenance, per-source stats, tokenizer fingerprint, data fingerprint
    <source>.<split>.tokens.bin   uint16/uint32 token stream; docs are <|bos|> ... <|eos|>
    <source>.<split>.mask.bin     uint8 per token: 1 = this token is a training target
    <source>.<split>.docs.bin     uint64 offset of each document start
"""

from __future__ import annotations

import glob
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from arouse.config import ConfigError
from arouse.data.cleaning import clean_text, dedup_key, is_validation
from arouse.data.config import DataConfig, DataSourceConfig
from arouse.tokenizer import ArouseTokenizer, Special

FORMAT = "arouse-packed-data"
FORMAT_VERSION = 1
SPLITS = ("train", "val")


def token_dtype(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint16) if vocab_size <= 65536 else np.dtype(np.uint32)


def source_files(src: DataSourceConfig) -> list[Path]:
    files: list[Path] = []
    for pat in src.paths:
        matches = sorted(glob.glob(pat, recursive=True))
        if not matches:
            raise ConfigError(f"source {src.name}: pattern matched no files: {pat}")
        files.extend(Path(m) for m in matches if Path(m).is_file())
    return files


def read_documents(src: DataSourceConfig, path: Path) -> Iterator[str]:
    if src.format == "jsonl":
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                value = obj.get(src.text_field)
                if not isinstance(value, str):
                    raise ConfigError(f"{path}:{n}: missing string field {src.text_field!r}")
                yield value
        return
    text = path.read_text(encoding="utf-8")
    if src.split == "file":
        yield text
    elif src.split == "paragraphs":
        yield from (p for p in text.replace("\r\n", "\n").split("\n\n") if p.strip())
    else:
        yield from (line for line in text.splitlines() if line.strip())


def assistant_mask(ids: list[int]) -> list[int]:
    """1 for tokens inside an Arouse turn (after <|arouse|>, up to and including its <|end|>)."""
    mask, inside = [], False
    for t in ids:
        if t == Special.AROUSE:
            mask.append(0)
            inside = True
        elif inside:
            mask.append(1)
            if t == Special.END:
                inside = False
        else:
            mask.append(0)
    return mask


def encode_document(tok: ArouseTokenizer, text: str, src: DataSourceConfig) -> tuple[list[int], list[int]]:
    ids = tok.encode(text, allow_special=src.allow_special)
    if not ids or ids[0] != Special.BOS:
        ids = [Special.BOS, *ids]
    if ids[-1] != Special.EOS:
        ids.append(Special.EOS)
    mask = [1] * len(ids) if src.loss_on == "all" else assistant_mask(ids)
    mask[0] = 0  # <|bos|> is never a target (nothing precedes it within the document)
    return ids, mask


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(cfg: DataConfig) -> dict[str, Any]:
    tok = ArouseTokenizer.load(cfg.tokenizer)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = token_dtype(tok.vocab_size)
    seen: dict[str, str] = {}  # dedup key -> source that kept it (global, first source wins)
    sources_meta = []

    for src in cfg.sources:
        buffers: dict[str, tuple[list[int], list[int]]] = {s: ([], []) for s in SPLITS}
        starts: dict[str, list[int]] = {s: [] for s in SPLITS}
        stats = {"docs_read": 0, "docs_dropped_clean": 0, "docs_dropped_dup": 0, "docs": {s: 0 for s in SPLITS}}
        files = source_files(src)
        for f in files:
            for raw in read_documents(src, f):
                stats["docs_read"] += 1
                text = clean_text(raw, min_chars=cfg.min_chars, max_bad_char_ratio=cfg.max_bad_char_ratio)
                if text is None:
                    stats["docs_dropped_clean"] += 1
                    continue
                key = dedup_key(text)
                if key in seen:
                    stats["docs_dropped_dup"] += 1
                    continue
                seen[key] = src.name
                split = "val" if is_validation(text, cfg.val_fraction) else "train"
                ids, mask = encode_document(tok, text, src)
                starts[split].append(len(buffers[split][0]))
                buffers[split][0].extend(ids)
                buffers[split][1].extend(mask)
                stats["docs"][split] += 1

        tokens = {}
        for split, (ids, mask) in buffers.items():
            np.asarray(ids, dtype=dtype).tofile(out / f"{src.name}.{split}.tokens.bin")
            np.asarray(mask, dtype=np.uint8).tofile(out / f"{src.name}.{split}.mask.bin")
            np.asarray(starts[split], dtype=np.uint64).tofile(out / f"{src.name}.{split}.docs.bin")
            tokens[split] = {"tokens": len(ids), "target_tokens": int(sum(mask))}
        sources_meta.append({
            **{k: getattr(src, k) for k in ("name", "category", "origin", "license", "weight", "loss_on", "allow_special", "window")},
            "files": [{"path": f.as_posix(), "sha256": _sha256(f)} for f in files],
            **stats,
            "splits": tokens,
        })

    meta: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "name": cfg.name,
        "token_dtype": dtype.name,
        "tokenizer_fingerprint": tok.fingerprint(),
        "vocab_size": tok.vocab_size,
        "config": cfg.to_dict(),
        "sources": sources_meta,
    }
    meta["data_fingerprint"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:16]
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta
