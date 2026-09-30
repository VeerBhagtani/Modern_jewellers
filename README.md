# Arouse AI

An independently trained language model for **reliable agentic task execution**: understand intent → plan → call tools with structured actions → observe → verify → finish, or ask/fail honestly.

- Own tokenizer, architecture, training pipeline, weights, inference engine, and benchmark.
- Trained from random initialization. It is **not** a wrapper around, or a fine-tune of, any external LLM.
- The only numerical framework is PyTorch (from Milestone 2).

```
USER / MDA → Arouse API → inference engine → Arouse model → structured action (tool_call | ask_user | finish | fail)
```

MDA (Modern Dairy Assistant) is a separate frontend and is not part of this repo.

## Status

**Milestone 1 of 7 is complete.** See [docs/roadmap.md](docs/roadmap.md).

| Component | State |
|---|---|
| Config system (strict, typed YAML) | ✅ |
| `ModelConfig`: v0.1 = 110.1M params (12L, d=768, 12 heads, SwiGLU 2048, RoPE, RMSNorm, tied) | ✅ config only; no model code yet |
| Byte-level BPE tokenizer with agent special tokens | ✅ |
| Tiny tokenizer corpus + manifest | ✅ |
| Model, training, inference, protocol, agent, benchmark, API | ⏳ later milestones |

No model has been trained. No benchmark results exist yet.

## Setup

```bash
pip install -e ".[dev]"
pytest -q
```

## Commands

```bash
arouse model info --config configs/model.yaml
arouse tokenizer train --config configs/tokenizer_tiny.yaml       # -> artifacts/tokenizers/arouse-tiny
arouse tokenizer encode -v --allow-special --tokenizer artifacts/tokenizers/arouse-tiny '<|user|>Remind me at 08:00<|end|>'
arouse tokenizer decode --tokenizer artifacts/tokenizers/arouse-tiny 9 3
arouse tokenizer stats  --tokenizer artifacts/tokenizers/arouse-tiny datasets/tokenizer_tiny/general.txt
python scripts/build_tiny_tokenizer_corpus.py                     # regenerate synthetic data + manifest
```

## Layout

```
configs/            model.yaml (v0.1), model_tiny.yaml, tokenizer.yaml, tokenizer_tiny.yaml
arouse/
  config.py         strict typed config base (unknown keys / wrong types fail loudly)
  cli.py            `arouse` command
  model/config.py   ModelConfig + exact parameter accounting + presets
  tokenizer/        special_tokens.py, pretokenize.py, bpe.py, tokenizer.py, trainer.py
  training/ inference/ agent/ protocol/ evaluation/    (later milestones)
datasets/           provenance-tracked data (MANIFEST.json per dataset)
docs/               tokenizer.md (design), roadmap.md
scripts/            dataset generators
tests/              pytest suite
checkpoints/ benchmarks/   (later; checkpoints are git-ignored)
```

## Scaling

The architecture is defined entirely by YAML. For example, `d_model: 1024, n_layers: 24, n_heads: 16, n_kv_heads: 8` gives about 350M parameters with no code changes (see `tests/test_config.py::test_scaling_is_config_only`).
