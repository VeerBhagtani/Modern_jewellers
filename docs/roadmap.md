# Roadmap

Each milestone is: explain → implement → test → run tests → fix → report. Nothing is marked done without passing tests.

| # | Milestone | Status |
|---|---|---|
| 1 | Project structure, config system, tokenizer design + implementation, tiny dataset, tests, docs | ✅ done |
| 2 | Transformer in PyTorch (RoPE, GQA, SwiGLU, RMSNorm, KV cache); param count equals config | ✅ done |
| 2b | Inference core (sampling, streaming, token safety mask), `arouse chat`, `arouse.generate()`, local API + chat UI | ✅ done |
| 3 | Dataset pipeline (provenance, cleaning, dedup, mixing, packing, splits) + resumable training | ✅ done |
| 4 | Arouse Action Protocol v1: schemas, validation, canonical JSON, token codec | ✅ done |
| 5 | Structured output: protocol validation, copy- and date-constrained decoding, schema-aware closing, grounding and answer guards, bounded resampling | ✅ done |
| 6 | Agent runtime (context, sandbox tools, task state, verification guard), agent data, agent model training | ✅ done |
| 7 | Arouse AgentBench v1 (held-out split) + `/v1/agent` API for MDA. Results in README / `benchmarks/results/` | ✅ done |
| 8 | Weak-spot round: `agent_v3` data + 3,000 more training steps, plus runtime fixes (dates follow the request, verbatim notes, whole file names, delete safety, confirmations from tool results). Held-out end-to-end success 71.25% → 92.75% | ✅ done |
| 9 | Website: the model runs in the browser (`web/`, JS port with parity tests), GitHub Pages workflow | ✅ done |
| 10 | Questions, GST and leads: knowledge base + `kb.search`, `gst.calculate`, `leads.find` (three methods or the user's own); agent_v4 data, 4,000 more steps at a 1,024-token context; AgentBench v2 with answer and whole-conversation metrics | ✅ done |

## Next

Measured weak spots (AgentBench v2 held-out, whole conversations):
- lead requests with unseen wording: 22 to 39%. "Who stopped buying from me?" or "number two please" often get the three-method question again
- deletes with unseen phrasing: 54%. "Please drop the … reminder" is read as a new reminder
- relative reminders with unseen phrasing: 71% ("Half an hour from now, …")

Planned work:
- A larger model (the 110M v0.1 config) on a GPU, with real, provenance-tracked pretraining text plus agent SFT data, so that wording generalizes beyond templates.
- A larger knowledge base, with dated entries for rules that change (GST rates).
- Lead sources beyond `customers.csv`, such as client-executed tools in MDA (web or CRM search).
- A fresh held-out benchmark: the current ones have informed runtime design.

## Hardware note

The dev container has **no GPU** (4 CPU cores).
- `arouse-agent-s` (5.5M params) was trained on CPU: 11,000 steps and about 135M tokens in total.
- The ~110M v0.1 configuration needs a CUDA GPU. The code path is the same, but no v0.1 training run has been done.
