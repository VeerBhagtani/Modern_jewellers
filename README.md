# Arouse AI

An independently trained language model for **reliable agentic task execution**: understand intent, plan, call tools with structured actions, observe the result, verify it, and then finish (or ask, or fail honestly).

- **Built from scratch:** own tokenizer, Transformer, training pipeline, weights, inference engine, agent runtime and benchmark.
- **Not a wrapper:** trained from random initialization. It is not a fine-tune of, or a proxy for, any external LLM.
- **Frameworks:** PyTorch is the only numerical framework. The API server uses the Python standard library.

```
USER / MDA → Arouse API → agent runtime → inference engine → Arouse model → structured action
                                  ↑                                              (tool_call | ask_user | finish | fail)
                          tools run here (sandbox, or MDA)
```

MDA (Modern Dairy Assistant) is a separate frontend. It connects through `POST /v1/agent` ([docs/api.md](docs/api.md)).

![Arouse chat UI](docs/images/chat_ui.png)

## Quick start (uses the trained model in this repo)

```bash
pip install -e ".[dev]"
arouse serve --model models/arouse-agent-s        # open http://127.0.0.1:8000
arouse agent --model models/arouse-agent-s        # same agent in the terminal
```

Try messages like these:
- "Remind me tomorrow at 8 AM to check sales."
- "Every Monday at 9 AM remind me to pay the staff."
- "Remind me on Friday to call the vet." (it asks for the time)
- "What reminders do I have?"
- "Cancel my reminder to check sales."
- "Note that the feed delivery is late."
- "How many lines are in sales.csv?"

Tools run in a local in-memory sandbox. A real, unedited conversation is in [docs/demo_transcript.md](docs/demo_transcript.md).

## What Arouse can and cannot do (honest scope)

`models/arouse-agent-s` has **5.5M parameters** and was **trained on CPU** in this repo:
- 2,000 steps on `agent_v1`
- then a 2,000-step fine-tune on `agent_v2`
- about 49M tokens in total

The training data is **synthetic agent episodes only**.

**It can:**
- create one-time, relative and recurring reminders (including follow-up questions for missing details)
- list and delete reminders
- save notes
- read and list files
- recover from tool errors
- handle greetings, thanks and help questions
- honestly refuse out-of-scope requests

**It cannot:**
- answer general-knowledge questions, write essays or code, or hold open conversation. It has never seen that kind of data. That requires the larger v0.1 model (110M), real pretraining text and a GPU.
- handle phrasing far from its training distribution reliably. The numbers below show how much this matters.

## Results (Arouse AgentBench v1, real runs)

Final benchmark results are being generated (`benchmarks/results/`).

## How reliability is engineered

| Layer | What it guarantees |
|---|---|
| Action protocol ([docs/protocol.md](docs/protocol.md)) | Exactly one schema-validated action per turn; a single token decides the action type |
| Token mask | The model cannot emit `<|user|>`, `<|tool_result|>` and similar tokens, so it cannot fake an observation |
| Copy-constrained decoding | Task, note and path text can only be copied from the user's words or listed files. Dates can only come from the runtime calendar or dates the user wrote. The model still chooses which phrase or date |
| Answer guard | Final answers may only use Arouse's trained response vocabulary plus words from the conversation or tool results |
| Validation, retries, grounding | Invalid or ungrounded turns are resampled; otherwise the runtime returns an honest `fail` |
| Completion guard | A `finish` right after a failed tool call becomes `fail`. False "Done." messages never reach the user |
| Deterministic tools | "The model understands, code computes": the scheduler turns rules and offsets into times |

## Build it yourself

```bash
python scripts/build_agent_data.py --name agent_v1                           # synthetic episodes + AgentBench files
arouse tokenizer train --config configs/tokenizer_agent.yaml
arouse data prepare --config configs/data_agent.yaml
arouse train --config configs/train_agent_s.yaml                             # auto-resumes after a crash
python scripts/build_agent_data.py --name agent_v2 && arouse data prepare --config configs/data_agent_v2.yaml
arouse train --config configs/train_agent_s_v2.yaml                          # fine-tune (init_from)
arouse model export --model checkpoints/arouse-agent-s-v2 --out models/arouse-agent-s
arouse bench --model models/arouse-agent-s --file benchmarks/agentbench_v1/test.jsonl
```

Notes:
- `train_agent_s.yaml` is the 4,000-step schedule; the shipped model used its step-2,000 checkpoint.
- The generator now produces the v2 format. The exact v1-phase data came from commit `246e0eb`.
- `models/arouse-agent-s/` contains the exact tokenizer the model was trained with.
- Rebuilding the data regenerates `benchmarks/agentbench_v1/*.jsonl` from fixed seeds, so the committed benchmark files are overwritten with identical content.

The 110M-parameter v0.1 configuration (`configs/model.yaml`, 110,119,680 params) uses the same code and needs a GPU.

## Milestones

All 7 are complete. See [docs/roadmap.md](docs/roadmap.md).

| # | Milestone | Docs |
|---|---|---|
| 1 | Tokenizer (byte-level BPE, 64 frozen special slots, agent tokens), config system | [docs/tokenizer.md](docs/tokenizer.md) |
| 2 | Transformer (RoPE, GQA, SwiGLU, RMSNorm, KV cache) | `arouse/model/` |
| 3 | Data pipeline (provenance, dedup, mixing, packing) + resumable training (bitwise-exact resume) | [docs/training.md](docs/training.md) |
| 4 | Action protocol + token codec + tool schemas | [docs/protocol.md](docs/protocol.md) |
| 5 | Inference engine + constrained decoding | `arouse/inference/`, `arouse/agent/constraints.py` |
| 6 | Agent runtime, sandbox tools, synthetic agent data, trained model | [docs/agent.md](docs/agent.md) |
| 7 | AgentBench + local API + chat UI | [docs/api.md](docs/api.md) |

## Layout

```
arouse/
  tokenizer/  model/  data/  training/  inference/
  protocol/   actions, tool schemas, token codec
  agent/      context (calendar), tools (sandbox), episode encoding, runtime, grounding, constraints, synth
  evaluation/ agentbench.py
  api/        server.py (stdlib HTTP), agent_service.py, static/index.html (chat UI)
configs/      model, tokenizer, data and training YAMLs
models/       arouse-agent-s (trained weights, 22 MB)
benchmarks/   agentbench_v1 cases + results/
datasets/     provenance manifests
docs/         design docs, demo transcript
tests/        pytest suite
```

## Tests

```bash
pytest -q
```

The suite includes a Playwright browser test of the UI, which skips itself if Chromium isn't installed.
