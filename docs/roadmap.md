# Roadmap

Each milestone is: explain → implement → test → run tests → fix → report. Nothing is marked done without passing tests.

| # | Milestone | Status |
|---|---|---|
| 1 | Project structure, config system, tokenizer design + implementation, tiny dataset, tests, docs | ✅ done |
| 2 | Transformer in PyTorch (RoPE, GQA, SwiGLU, RMSNorm, KV cache); param count equals config | ✅ done |
| 2b | Inference core (sampling, streaming, token safety mask), `arouse chat`, `arouse.generate()`, local API + chat UI | ✅ done |
| 3 | Dataset pipeline (provenance, cleaning, dedup, mixing, packing, splits) + resumable training | ✅ done |
| 4 | Arouse Action Protocol v1: schemas, validation, canonical JSON, token codec | ✅ done |
| 5 | Structured-output mode: protocol validation + bounded resampling (grammar-constrained decoding: future) | ✅ done |
| 6 | Agent runtime (context, sandbox tools, task state, verification guard), agent data, agent model training | ✅ done |
| 7 | Arouse AgentBench v1 (held-out split) + `/v1/agent` API for MDA | ✅ done |

## Next

- A larger model (the 110M v0.1 config) on a GPU, with real, provenance-tracked pretraining text plus agent SFT data.
- More tools and tool-set variation (so the model conditions on `<|tools|>`), plus memory retrieval.
- Grammar-constrained JSON decoding, and paraphrase robustness (typos, code-mixed Hindi/English).

## Hardware note

The dev container has **no GPU** (4 CPU cores). The tiny model and pipeline tests run on CPU. Pretraining the ~110M v0.1 model needs a CUDA GPU; the code will support it, but no training run will be claimed until one actually happens.
