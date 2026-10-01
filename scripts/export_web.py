#!/usr/bin/env python3
"""Export a trained Arouse model for the in-browser website (web/).

    web/model/config.json     architecture, tensor table, provenance
    web/model/weights.bin     all weights as float16, little-endian, in table order
    web/model/tokenizer.json  special tokens + BPE merges (the exact tokenizer the model was trained with)
    web/model/response_vocab.txt
    web/model/knowledge.json  the knowledge base + its search word lists

Usage: python scripts/export_web.py --model models/arouse-agent-s --out web/model
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from arouse.agent import business, knowledge
from arouse.model.io import load_pretrained


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/arouse-agent-s")
    ap.add_argument("--out", default="web/model")
    a = ap.parse_args()
    model, tok, meta = load_pretrained(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = model.config
    state = model.state_dict()
    names = ["embed.weight"]
    for i in range(cfg.n_layers):
        names += [f"layers.{i}.{n}" for n in ("attn_norm.weight", "attn.wq.weight", "attn.wk.weight", "attn.wv.weight",
                                                "attn.wo.weight", "mlp_norm.weight", "mlp.w_gate.weight", "mlp.w_up.weight",
                                                "mlp.w_down.weight")]
    names.append("norm.weight")
    if not cfg.tie_embeddings:
        names.append("lm_head.weight")
    table, chunks, offset = [], [], 0
    for n in names:
        t = state[n].detach().float().cpu().numpy()
        table.append({"name": n, "shape": list(t.shape), "offset": offset})
        chunks.append(t.astype("<f2").ravel())
        offset += t.size
    weights = np.concatenate(chunks)
    weights.tofile(out / "weights.bin")
    config = {
        "format": "arouse-web-model", "format_version": 1,
        **{k: getattr(cfg, k) for k in ("name", "vocab_size", "context_length", "d_model", "n_layers", "n_heads",
                                         "rope_theta", "norm_eps", "tie_embeddings")},
        "n_kv_heads": cfg.kv_heads, "ffn_hidden_size": cfg.ffn_size,
        "num_parameters": model.num_parameters(), "train_steps": meta.get("train_steps", 0), "notes": meta.get("notes", ""),
        "tokenizer_fingerprint": tok.fingerprint(), "dtype": "float16", "tensors": table,
    }
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    tok_spec = {"special_tokens": tok._special_texts, "merges": [list(m) for m in tok.merges], "vocab_size": tok.vocab_size,
                "fingerprint": tok.fingerprint()}
    (out / "tokenizer.json").write_text(json.dumps(tok_spec, separators=(",", ":")) + "\n", encoding="utf-8")
    shutil.copy(Path("arouse/agent/response_vocab.txt"), out / "response_vocab.txt")
    # the knowledge base plus the exact word lists its search uses (web/arouse.js reads them from here)
    kb = json.loads(Path("arouse/agent/knowledge.json").read_text(encoding="utf-8"))
    kb["search"] = {"stopwords": sorted(knowledge.STOPWORDS), "synonyms": knowledge.SYNONYMS, "no_stem": sorted(knowledge.NO_STEM),
                    "lead_stopwords": sorted(business.LEAD_STOPWORDS)}
    (out / "knowledge.json").write_text(json.dumps(kb, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"exported {cfg.name}: {weights.size:,} weights ({weights.nbytes / 1e6:.1f} MB fp16) -> {out}")


if __name__ == "__main__":
    main()
