# Agent Runtime, Data and Benchmark

## Runtime loop (`arouse/agent/runtime.py`)

```
user message → prompt (header + events) → model turn → decode + validate → tool call? → execute → observation → repeat
                                                                         └→ ask_user / finish / fail → return to user
```

Layers of protection against unreliable model output:

| # | Protection | Where |
|---|---|---|
| 1 | Token mask: the model can't emit `<|user|>`, `<|tool_result|>`, `<|tool_error|>`, `<|state|>`, … so it can't forge an observation | `inference/engine.py` |
| 2 | Protocol and schema validation: a turn must decode to one valid action, and tool arguments must match the tool's schema. Otherwise the turn is resampled (greedy first, then 2 sampled retries), and finally an honest `fail` | `runtime.next_turn` |
| 2b | Copy-constrained decoding. The model still chooses what to copy:<br>• `task`: only continuations of the user's words<br>• `text` and `query`: only continuations of the user's words, verbatim to the end of the sentence; for "label: content" messages, the whole content<br>• `path`: whole file names the user wrote or a tool listed<br>• `date`: only the dates the request refers to<br>• `time`: only clock times the user wrote ("6:40pm", "9 in the morning"; "at 7" allows 07:00 or 19:00)<br>• `amount` and `rate` (GST): money phrases and %-rates exactly as written | `agent/constraints.py`, `agent/mentions.py` |
| 2c | Grounding check, which triggers resampling:<br>• any remaining ungrounded argument<br>• a `scheduler.delete` whose reminder was not listed, or is not the listed reminder that best matches the user's words | `agent/grounding.py` |
| 2d | Answer guard: a final answer may only use Arouse's reply vocabulary plus words from this request. Some text is written from the tool results instead of by the model:<br>• a confirmation after a successful tool call (reminder set, GST figures, knowledge-base answer, lead list)<br>• the "which file?" question after a missing file<br>• the answer, if no sample passes the guard<br>If no grounded delete is found, the runtime fails instead of deleting | `runtime.next_turn`, `agent/answers.py` |
| 3 | Completion guard: a `finish` that directly follows a failed tool call, with no success since, is rewritten to `fail` | `runtime._guard` |
| 4 | Step limit: at most 6 tool calls per user message | `runtime.run` |
| 5 | Sandbox tools never raise. Bad input comes back as a `tool_error` observation the agent must handle | `agent/tools.py` |

**Task state:**
- The conversation events *are* the state. Each turn continues from them instead of restarting.
- `Header.state` (`<|state|>`) and `Header.memory` (`<|memory|>`) let a runtime inject a task-state object or retrieved long-term memory.
- When the context is full, the oldest whole exchanges are dropped first.

## Training data (`arouse/agent/synth.py`, `scripts/build_agent_data.py`)

- Episodes are generated from scenarios, and **every tool result comes from actually executing the gold call in the sandbox**.
- Scenarios cover:
  - one-time reminders (tomorrow, weekday names, explicit dates, "today" that has already passed)
  - time-only and relative reminders
  - recurring reminders (weekly, weekday pairs, weekdays, weekends, daily, every N days, monthly)
  - missing time or missing task → ask, then continue once the user answers
  - list, delete by name
  - notes
  - file read, file count, missing file → list → "did you mean"
  - transient errors → retry once → success or honest fail
  - greetings, help, thanks, and honest refusal of out-of-scope requests
  - questions answered from the knowledge base (or "I don't know that yet"), GST calculations, finding leads ([skills.md](skills.md))
  - multi-request conversations
- **Held-out split:** the benchmark uses tasks, note facts, file names and sentence templates that never appear in training (tested). The test split is byte-for-byte frozen: generator changes that add training variety must not change it (tested by sha256).
- **v3 data** adds train-only variety:
  - more phrasings for one-time dates, notes, deletes and file questions
  - "how are you", goodbyes and acknowledgements
  - compositional out-of-scope questions
  - polite prefixes and suffixes ("hey, …", "… thanks")
- Training tasks are partly compositional (thousands of verb/object/person combinations and made-up names), so the model learns to **copy** what the user said instead of memorizing a list.
- The loss is computed only on Arouse's turns (`loss_on: assistant`). Training windows start at episode boundaries (`window: doc_start`).

## Arouse AgentBench (`arouse/evaluation/agentbench.py`)

```bash
arouse bench --model checkpoints/arouse-agent-s --file benchmarks/agentbench_v1/test.jsonl --out benchmarks/results/agent_s_test.json
```

- **Decision metrics** (raw model: greedy, no retries, no guard) are measured at every gold decision point, with the correct history up to that point:
  - structured-output validity
  - action-type accuracy
  - tool-selection accuracy
  - exact argument match
  - scheduling exact match
  - ask rate on ambiguous requests
  - **false-completion rate** (`finish` right after a tool error)
  - per-skill accuracy over 10 skills: intent, tool selection, argument generation, scheduling, multi-step, task state, error recovery, ambiguity, completion verification, structured output
- **End-to-end** (full runtime) runs each episode's first request against a fresh sandbox. It succeeds only if the terminal action type **and** the resulting reminders and notes match the gold outcome exactly.
- **Scorer sanity:** an oracle that replays the gold turns scores 100%, and a model that always says "Done." is penalized (`tests/test_benchmark.py`).
