# Arouse AI

An independently trained language model for **reliable agentic task execution**: understand intent → plan → call tools with structured actions → observe → verify → finish, or ask/fail honestly.

- Own tokenizer, architecture, training pipeline, weights, inference engine, and benchmark.
- Trained from random initialization. It is **not** a wrapper around, or a fine-tune of, any external LLM.
- The only numerical framework is PyTorch.

```
USER / MDA → Arouse API → inference engine → Arouse model → structured action (tool_call | ask_user | finish | fail)
```

MDA (Modern Dairy Assistant) is a separate frontend and is not part of this repo.

## Status

**Milestones 1–2 are complete, plus an early inference engine, API and chat UI.** See [docs/roadmap.md](docs/roadmap.md).

| Component | State |
|---|---|
| Config system (strict, typed YAML) | ✅ |
| Byte-level BPE tokenizer with agent special tokens | ✅ |
| Transformer (RoPE, GQA, SwiGLU, RMSNorm, KV cache): v0.1 = 110,119,680 params | ✅ |
| Inference engine (sampling, streaming, stop tokens, token safety mask) | ✅ core |
| Local API and chat UI ([docs/api.md](docs/api.md)) | ✅ `/v1/chat`, `/v1/generate`, `/v1/health` |
| Dataset pipeline and training | ⏳ Milestone 3 |
| Protocol, agent runtime, benchmark | ⏳ later |

**No model has been trained yet.** The chat UI currently runs a randomly initialised model, so its replies are noise, and the UI says so. There are no benchmark results yet.

## Setup

```bash
pip install -e ".[dev]"       # needs PyTorch; CPU is fine for the tiny model
pytest -q
```

## Chat UI (quick start)

```bash
arouse tokenizer train --config configs/tokenizer_tiny.yaml
arouse model init --config configs/model_tiny.yaml --tokenizer artifacts/tokenizers/arouse-tiny --out artifacts/models/arouse-tiny-random
arouse serve --model artifacts/models/arouse-tiny-random      # open http://127.0.0.1:8000
arouse chat  --model artifacts/models/arouse-tiny-random      # or chat in the terminal
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
  tokenizer/        special_tokens.py, pretokenize.py, bpe.py, tokenizer.py, trainer.py
  model/            config.py, embeddings.py (token + RoPE), attention.py, mlp.py, norm.py,
                    transformer_block.py, transformer.py, lm_head.py, loss.py, io.py (save/load)
  inference/        sampling.py, chat.py (prompt format), engine.py (KV-cached streaming)
  api/              server.py (stdlib HTTP), static/index.html (chat UI)
  training/ agent/ protocol/ evaluation/    (later milestones)
datasets/           provenance-tracked data (MANIFEST.json per dataset)
docs/               tokenizer.md (design), api.md (HTTP API), roadmap.md
scripts/            dataset generators
tests/              pytest suite
checkpoints/ benchmarks/   (later; checkpoints are git-ignored)
```

## Scaling

The architecture is defined entirely by YAML. For example, `d_model: 1024, n_layers: 24, n_heads: 16, n_kv_heads: 8` gives about 350M parameters with no code changes (see `tests/test_config.py::test_scaling_is_config_only`).
