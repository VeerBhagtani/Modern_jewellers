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

## Use it in your browser

`web/` is a static website that runs the trained model entirely in the browser, with no server-side model and no external AI service ([docs/web.md](docs/web.md)).

**Put it online with GitHub Pages:**
1. On GitHub, open the repository → **Settings** → **Pages**.
2. Under **Build and deployment → Source**, choose **GitHub Actions**.
3. Open the **Actions** tab → **Deploy website** → **Run workflow**.

The site appears at `https://veerbhagtani.github.io/Modern_jewellers/` and redeploys on every push that changes `web/`.

**On your own computer:** `python -m http.server 8080 -d web`, then open http://localhost:8080. Opening `index.html` directly does not work, because browsers block the model files on `file://`.

- **Workspace:** reminders, notes and files live in the browser. Four sample files are included, among them a `customers.csv` for finding leads, and you can add your own text or CSV files.
- **Firing reminders:** reminders fire while the page is open.
- **JS port:** the JavaScript runtime (`web/arouse.js`) is tested against the Python one:
  - same token ids and prompts
  - logits equal within float rounding
  - identical tool results, knowledge-base answers and decisions whenever no resampling is needed

![Arouse website](docs/images/web_ui.png)

## Quick start (Python, local API)

```bash
pip install -e ".[dev]"
arouse serve --model models/arouse-agent-s        # open http://127.0.0.1:8000
arouse agent --model models/arouse-agent-s        # same agent in the terminal
```

Try messages like these:
- "What is the GST on gold?" / "What is HUID?" / "How do I calculate profit margin?"
- "Calculate GST on 2.5 lakh at 3%." / "₹1,180 includes 18% GST. How much is the GST?"
- "Find me leads" (it offers three methods, or uses your own idea)
- "Remind me tomorrow at 8 AM to check sales."
- "Every Monday at 9 AM remind me to pay the staff."
- "Remind me on Friday to call the vet." (it asks for the time)
- "What reminders do I have?"
- "Cancel my reminder to check sales."
- "Note that the feed delivery is late."
- "How many lines are in sales.csv?"

Tools run in a local in-memory sandbox. A real, unedited conversation is in [docs/demo_transcript.md](docs/demo_transcript.md).

## What Arouse can and cannot do (honest scope)

`models/arouse-agent-s` has **5.5M parameters** and was **trained on CPU** in this repo, about 135M tokens in total:
- 2,000 steps on `agent_v1`
- 2,000 steps on `agent_v2`
- 3,000 steps on `agent_v3`, which adds phrasing variety aimed at the measured weak spots
- 4,000 steps on `agent_v4` at a 1,024-token context, which adds questions, GST and leads

The training data is **synthetic agent episodes plus a hand-written knowledge base** ([docs/skills.md](docs/skills.md)).

**It can:**
- answer easy questions about **GST, gold and jewellery, dairy and running a small business** from a built-in, source-checked knowledge base, and say "I don't know that yet" otherwise
- **calculate GST** on amounts as people write them ("2.5 lakh", "Rs. 12,500", inclusive prices), asking for the rate if it is missing
- **find leads** in your customer list. Arouse offers three methods, or uses your own idea:
  1. past customers to win back
  2. birthdays and anniversaries
  3. top customers to ask for referrals
- create one-time, relative and recurring reminders (including follow-up questions for missing details)
- list and delete reminders, save notes, read and list files
- recover from tool errors
- handle greetings, thanks and help questions, and honestly refuse requests it cannot do

**It cannot:**
- answer general-knowledge questions outside its knowledge base, write essays or code, or hold open conversation. It has never seen that kind of data. That requires the larger v0.1 model (110M), real pretraining text and a GPU.
- give live prices or news (it works offline), or find leads outside your own customer list (it has no internet access).
- handle phrasing far from its training distribution reliably. The numbers below show how much this matters.

## Results (Arouse AgentBench, real runs)

`arouse bench`, full runs, with reports in `benchmarks/results/` (earlier runs in `benchmarks/results/history/`).

**AgentBench v2** covers everything, including questions, GST and leads:
- **Held-out:** 1,416 decisions from 480 episodes. Their tasks, note facts, files, cities, interests, questions and sentence templates never appear in training.
- **In-distribution:** 523 decisions from 180 episodes. Same templates as training, but unseen samples.

Modes and metrics:
- **Raw:** the model alone (greedy decoding).
- **System:** the full runtime (constrained decoding, retries, grounding, guards).
- **End-to-end:** the first request, scored on the final action type and the resulting reminders and notes.
- **Exact answer:** for answers that state a tool result (GST figures, knowledge-base answers, lead lists, created reminders), the text must match exactly.
- **Whole conversation:** every user message of the episode in turn (for example "Find me leads" → "the second one"). All replies and the final state must be right.

| Metric | Held-out raw | Held-out system | In-dist raw | In-dist system |
|---|---|---|---|---|
| Decision accuracy (type + exact tool and arguments) | 68.29% | **93.08%** | 93.31% | **100.0%** |
| Structured-output validity | 93.5% | **99.72%** | 99.81% | **100.0%** |
| Tool-selection accuracy | 86.97% | **90.58%** | 100.0% | **100.0%** |
| Argument exact match | 43.01% | **89.64%** | 86.03% | **100.0%** |
| Asks when a detail is missing | 83.7% | **84.44%** | 100.0% | **100.0%** |
| False completion (finish after a tool error) | 0.0% | **0.0%** | 0.0% | **0.0%** |
| **End-to-end task success** | — | **90.83%** | — | **100.0%** |
| **Exact answer** (319 / 120 answers) | — | **89.66%** | — | **99.17%** |
| **Whole conversations right** | — | **83.12%** | — | **99.44%** |

**Held-out, by skill (system, end to end):**

| Skill | End to end | Whole conversation |
|---|---|---|
| Knowledge questions (39) | **100%**, exact answer 100% | 100% |
| Unknown questions → "I don't know that yet" (23) | **100%** | 90.3% |
| GST calculation (34) | **100%**, exact figures 100% | 81.3% |
| Notes (18), files (30), recurring reminders (39) | **100%** | 88 to 93% |
| One-time / relative reminders (63 / 18) | 90.5% / 88.9% | 83.3% / 70.8% |
| Leads: user picks a method (18) / own idea (6) | 77.8% / 83.3% | **29.2% / 22.2%** |
| Leads: method named directly (19) | **36.8%** | 39.1% |
| Delete (23) | **60.9%** | 54.1% |

**AgentBench v1** (frozen, reminders and notes only, 400 held-out episodes) shows the new skills did not cost the old ones. End to end:

| Model | Held-out, end to end | In-distribution, end to end |
|---|---|---|
| Previous release | 92.75% | 96.67% |
| This model | **93.25%** | **99.33%** |

By category:
- Delete improved: 42.9% → 64.3%.
- Ambiguous requests dropped: 94.1% → 91.2%.
- Answering a follow-up dropped: 96.9% → 90.6%.

**Read these numbers with care:**
- **Unseen wordings for leads are the weakest area.** "Who stopped buying from me?" or "number two please" often gets the three-method question again, or is treated as a knowledge question. Wordings close to training work (100% in-distribution).
- **The runtime rules were designed after inspecting held-out failures, so the held-out sets are not fully blind for the runtime.** These rules: dates, times, amounts and rates must come from the user's words; whole file names only; answers stated from tool results. The rules are general (no template strings), and the raw-model numbers are not affected.
- **Knowledge search on its own:** 27 of 33 held-out paraphrased questions find the right entry. The other six get "I don't know that yet" (five) or a wrong entry (one).
- **Raw vs held-out:** the raw model alone gets 68% of held-out decisions right; the runtime brings it to 93%. A 5.5M model trained on synthetic templates generalizes only partly to new phrasings. A larger model and real text are the next step.

## How reliability is engineered

| Layer | What it guarantees |
|---|---|
| Action protocol ([docs/protocol.md](docs/protocol.md)) | Exactly one schema-validated action per turn; a single token decides the action type |
| Token mask | The model cannot emit `<|user|>`, `<|tool_result|>` and similar tokens, so it cannot fake an observation |
| Copy-constrained decoding | The model still chooses which phrase or value:<br>• task, note and question text can only be copied from the user's words; notes are copied verbatim to the end of the sentence<br>• file paths must be whole file names the user wrote or a tool listed<br>• dates must be ones the request refers to; clock times must be times the user wrote<br>• GST amounts and rates must be exactly as the user wrote them |
| Delete safety | A reminder is deleted only if it was listed and is the listed reminder that best matches the user's words. Otherwise Arouse fails without deleting anything |
| Answer guard | Final answers may only use Arouse's trained response vocabulary plus words from the conversation or tool results. Answers that state a tool result (GST figures, knowledge-base answers, lead lists, confirmations, "which file?" questions) are checked against it and written from it if they differ. "Couldn't find" claims must agree with the listed reminders |
| Validation, retries, grounding | Invalid or ungrounded turns are resampled; otherwise the runtime returns an honest `fail` |
| Completion guard | A `finish` right after a failed tool call becomes `fail`. False "Done." messages never reach the user |
| Deterministic tools | "The model understands, code computes":<br>• the scheduler turns rules and offsets into times<br>• GST is computed in integer paise<br>• knowledge search and lead filters are deterministic |

## Build it yourself

```bash
python scripts/build_agent_data.py --name agent_v1                           # synthetic episodes + AgentBench files
arouse tokenizer train --config configs/tokenizer_agent.yaml
arouse data prepare --config configs/data_agent.yaml
arouse train --config configs/train_agent_s.yaml                             # auto-resumes after a crash
python scripts/build_agent_data.py --name agent_v2 && arouse data prepare --config configs/data_agent_v2.yaml
arouse train --config configs/train_agent_s_v2.yaml                          # fine-tune (init_from)
python scripts/build_agent_data.py --name agent_v3 && arouse data prepare --config configs/data_agent_v3.yaml
arouse train --config configs/train_agent_s_v3.yaml                          # fine-tune from the v2 model
python scripts/build_agent_data.py --name agent_v4 && arouse data prepare --config configs/data_agent_v4.yaml
arouse train --config configs/train_agent_s_v4.yaml                          # fine-tune from v3 at a 1024-token context
arouse model export --model checkpoints/arouse-agent-s-v4 --out models/arouse-agent-s
arouse bench --model models/arouse-agent-s --file benchmarks/agentbench_v2/test.jsonl
python scripts/export_web.py --model models/arouse-agent-s --out web/model   # float16 weights for the website
```

Notes:
- `train_agent_s.yaml` is the 4,000-step schedule; the shipped model used its step-2,000 checkpoint.
- The generator now produces the v4 data. The exact v1-phase data came from commit `246e0eb`; v3 data from commit `8f60c3d`.
- `models/arouse-agent-s/` contains the exact tokenizer the model was trained with.
- Rebuilding the data leaves the benchmark files alone unless you pass `--bench`. AgentBench v2's held-out split is frozen byte for byte (tested); v1 is kept on disk, frozen by its manifest hash.

The 110M-parameter v0.1 configuration (`configs/model.yaml`, 110,119,680 params) uses the same code and needs a GPU.

## Milestones

All 9 are complete. See [docs/roadmap.md](docs/roadmap.md).

| # | Milestone | Docs |
|---|---|---|
| 1 | Tokenizer (byte-level BPE, 64 frozen special slots, agent tokens), config system | [docs/tokenizer.md](docs/tokenizer.md) |
| 2 | Transformer (RoPE, GQA, SwiGLU, RMSNorm, KV cache) | `arouse/model/` |
| 3 | Data pipeline (provenance, dedup, mixing, packing) + resumable training (bitwise-exact resume) | [docs/training.md](docs/training.md) |
| 4 | Action protocol + token codec + tool schemas | [docs/protocol.md](docs/protocol.md) |
| 5 | Inference engine + constrained decoding | `arouse/inference/`, `arouse/agent/constraints.py` |
| 6 | Agent runtime, sandbox tools, synthetic agent data, trained model | [docs/agent.md](docs/agent.md) |
| 7 | AgentBench + local API + chat UI | [docs/api.md](docs/api.md) |
| 8 | Weak-spot round (agent_v3 + runtime fixes) | [docs/agent.md](docs/agent.md) |
| 9 | Website (in-browser model) + questions, GST and leads (agent_v4, AgentBench v2) | [docs/web.md](docs/web.md), [docs/skills.md](docs/skills.md) |

## Layout

```
arouse/
  tokenizer/  model/  data/  training/  inference/
  protocol/   actions, tool schemas, token codec
  agent/      context (calendar), tools (sandbox), episode encoding, runtime, grounding, constraints, synth,
              knowledge (knowledge.json + search), business (GST, leads), mentions, answers
  evaluation/ agentbench.py
  api/        server.py (stdlib HTTP), agent_service.py, static/index.html (chat UI)
web/          the website: arouse.js (JS port), worker.js, index.html, model/ (float16 weights, 11 MB)
configs/      model, tokenizer, data and training YAMLs
models/       arouse-agent-s (trained weights, 22 MB)
benchmarks/   agentbench_v1 (frozen) and agentbench_v2 cases + results/
datasets/     provenance manifests
docs/         design docs, demo transcript
tests/        pytest suite
```

## Tests

```bash
pytest -q
```

The suite includes:
- a Playwright browser test of the UI, which skips itself if Chromium isn't installed
- JS-vs-Python parity tests for the website, which skip themselves if Node isn't installed
