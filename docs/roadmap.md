# Roadmap

Each milestone is: explain → implement → test → run tests → fix → report. Nothing is marked done without passing tests.

| # | Milestone | Status |
|---|---|---|
| 1 | Project structure, config system, tokenizer design + implementation, tiny dataset, tests, docs | ✅ done |
| 2 | Transformer in PyTorch (embeddings, RoPE, attention + causal mask, SwiGLU, RMSNorm, LM head, loss); param count must equal `ModelConfig.parameter_counts()` | next |
| 3 | Dataset pipeline (sources + provenance, cleaning, dedup, mixing weights, packing, splits) + resumable pretraining (AMP, grad accumulation, LR schedule, checkpoints) | |
| 4 | Arouse Action Protocol: versioned JSON schemas, validator, `<|X|>body` ↔ `{"type":X,…}` codec, constrained decoding | |
| 5 | Inference engine (KV cache, sampling, stop tokens, streaming, structured mode), `arouse chat`, `arouse.generate()` | |
| 6 | Agent runtime: task state, memory injection, verification, trajectory format + agent SFT data (scheduling, errors, clarification) | |
| 7 | Arouse AgentBench (≥500 cases, held-out split) + local HTTP API (`/v1/generate`, `/v1/chat`, `/v1/agent`, `/v1/health`) | |

## Hardware note

The dev container has **no GPU** (4 CPU cores). The tiny model and pipeline tests run on CPU. Pretraining the ~110M v0.1 model needs a CUDA GPU; the code will support it, but no training run will be claimed until one actually happens.
