"""`arouse` command-line interface.

    arouse model info --config configs/model.yaml
    arouse tokenizer train --config configs/tokenizer_tiny.yaml
    arouse tokenizer encode --tokenizer DIR [--allow-special] TEXT
    arouse tokenizer decode --tokenizer DIR ID [ID ...]
    arouse tokenizer stats --tokenizer DIR FILE [FILE ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from arouse.model.config import ModelConfig
from arouse.tokenizer import ArouseTokenizer, TokenizerTrainingConfig, train_tokenizer
from arouse.tokenizer.trainer import iter_documents


def _model_info(a: argparse.Namespace) -> int:
    cfg = ModelConfig.from_yaml(a.config)
    print(cfg.summary())
    print(json.dumps(cfg.parameter_counts(), indent=2))
    return 0


def _tok_train(a: argparse.Namespace) -> int:
    cfg = TokenizerTrainingConfig.from_yaml(a.config)
    if a.output_dir:
        cfg = cfg.replace(output_dir=a.output_dir)
    t0 = time.perf_counter()
    tok = train_tokenizer(cfg)
    out = tok.save(cfg.output_dir)
    info = tok.training_info
    print(
        f"trained vocab={tok.vocab_size} (target {cfg.vocab_size}) merges={len(tok.merges)} "
        f"bytes={info['bytes_used']} chunks={info['unique_chunks']} "
        f"time={time.perf_counter() - t0:.1f}s fingerprint={tok.fingerprint()} -> {out}"
    )
    if tok.vocab_size < cfg.vocab_size:
        print("note: corpus too small to reach target vocab (min_frequency stop)")
    return 0


def _tok_encode(a: argparse.Namespace) -> int:
    tok = ArouseTokenizer.load(a.tokenizer)
    ids = tok.encode(a.text, allow_special=a.allow_special)
    print(json.dumps(ids))
    if a.verbose:
        for i in ids:
            print(f"{i:>6}  {tok.id_to_bytes(i)!r}")
    return 0


def _tok_decode(a: argparse.Namespace) -> int:
    tok = ArouseTokenizer.load(a.tokenizer)
    print(tok.decode(a.ids))
    return 0


def _tok_stats(a: argparse.Namespace) -> int:
    tok = ArouseTokenizer.load(a.tokenizer)
    n_bytes = n_tokens = 0
    for f in a.files:
        for doc in iter_documents(Path(f)):
            n_bytes += len(doc.encode("utf-8"))
            n_tokens += len(tok.encode(doc, allow_special=True))
    ratio = n_bytes / n_tokens if n_tokens else 0.0
    print(f"bytes={n_bytes} tokens={n_tokens} bytes/token={ratio:.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arouse", description="Arouse AI core tools")
    sub = p.add_subparsers(dest="group", required=True)

    model = sub.add_parser("model").add_subparsers(dest="cmd", required=True)
    info = model.add_parser("info", help="show architecture + parameter count")
    info.add_argument("--config", default="configs/model.yaml")
    info.set_defaults(fn=_model_info)

    tok = sub.add_parser("tokenizer").add_subparsers(dest="cmd", required=True)
    tr = tok.add_parser("train", help="train a tokenizer from a YAML config")
    tr.add_argument("--config", required=True)
    tr.add_argument("--output-dir")
    tr.set_defaults(fn=_tok_train)

    enc = tok.add_parser("encode")
    enc.add_argument("--tokenizer", required=True)
    enc.add_argument("--allow-special", action="store_true")
    enc.add_argument("-v", "--verbose", action="store_true")
    enc.add_argument("text")
    enc.set_defaults(fn=_tok_encode)

    dec = tok.add_parser("decode")
    dec.add_argument("--tokenizer", required=True)
    dec.add_argument("ids", nargs="+", type=int)
    dec.set_defaults(fn=_tok_decode)

    st = tok.add_parser("stats", help="compression on .txt/.jsonl files")
    st.add_argument("--tokenizer", required=True)
    st.add_argument("files", nargs="+")
    st.set_defaults(fn=_tok_stats)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
