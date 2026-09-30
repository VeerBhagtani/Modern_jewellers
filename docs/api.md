# Arouse Local API (v1)

```bash
arouse serve --model <model_dir> [--host 127.0.0.1] [--port 8000]
```

The server uses only the Python standard library. It binds to localhost by default and runs one generation at a time (requests queue behind a lock). The chat UI is served at `/`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Chat UI (agent mode) |
| GET | `/v1/health` | Status and model info |
| POST | `/v1/agent` | **Next structured action** for a conversation the client manages (for MDA) |
| POST | `/v1/agent/run` | Local demo: server session plus sandbox tools; runs until a terminal action |
| POST | `/v1/agent/reset` | Forget a demo session |
| POST | `/v1/chat` | Raw chat completion (plain text, no tools) |
| POST | `/v1/generate` | Raw text completion |

## Agent endpoints

### `POST /v1/agent`: MDA integration (stateless)

MDA owns the conversation and executes tools itself. Each call returns exactly one validated action.

```json
{"events": [
   {"type": "user", "content": "Every Monday at 9 AM remind me to check sales."}
 ],
 "context": {"now": "2026-09-30T19:21", "...": "optional, built from server time if omitted"},
 "tools": ["scheduler.create", "scheduler.list"],
 "memory": "optional retrieved memory text",
 "state": {"optional": "task state object"}}
```

```json
{"protocol": "arouse-action/1",
 "plan": "Recurring reminder every Monday at 09:00.",
 "action": {"type": "tool_call", "tool": "scheduler.create",
            "arguments": {"repeat": {"by_day": ["MO"], "freq": "weekly"}, "task": "check sales", "time": "09:00"}},
 "valid": true, "attempts": 1, "guarded": false}
```

After executing the call, MDA sends the same request again with two more events appended:
- the turn it just received: `{"type": "arouse", "turn": {...}}`
- the observation: `{"type": "tool_result", "content": {...}}` or `{"type": "tool_error", "content": {...}}`

Stop when the returned action is `ask_user`, `finish` or `fail`.

In the response:
- `valid: false` means the model produced no valid action. The action is then an honest `fail`.
- `guarded: true` means a `finish` that followed a failed tool call was rewritten to `fail`.

### `POST /v1/agent/run`: built-in sandbox (chat UI)

Request: `{"message": "...", "session_id": "optional"}`.

Response: `{"session_id", "final": action, "steps": [turns and observations], "valid", "guarded", "state": {"reminders", "notes"}}`.

The sandbox is in memory, with a few demo files (`sales.csv`, `milk_records.csv`, `notes.txt`, `staff.csv`).

### Sampling fields (chat and generate)

| Field | Default | Rule |
|---|---|---|
| `max_tokens` | 256 | ≥ 1 |
| `temperature` | 0.8 | ≥ 0 (0 = greedy) |
| `top_k` | 0 | ≥ 0 (0 = off) |
| `top_p` | 1.0 | in (0, 1] |
| `seed` | null | int, for reproducible sampling |
| `stream` | false | Server-Sent Events |

Unknown fields are rejected with `400`, so typos never silently fall back to defaults. `model` is accepted and ignored, since each server runs one model.

### `POST /v1/chat`

```json
{"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}], "max_tokens": 100}
```

Roles are `system`, `user` and `assistant`. When the conversation is too long, the oldest non-system messages are dropped; the system prompt and the latest message are always kept.

```json
{"id": "arouse-…", "created": 1790797796, "model": "arouse-tiny",
 "message": {"role": "assistant", "content": "…"},
 "finish_reason": "stop | length | context",
 "usage": {"prompt_tokens": 6, "completion_tokens": 8}}
```

With `"stream": true`, the response is a series of events:

```
data: {"delta": "text"}
…
data: {"done": true, "finish_reason": "stop", "usage": {…}}
```

Closing the connection stops generation and frees the model.

### `POST /v1/generate`

Body: `{"prompt": "...", "allow_special": false, ...sampling}`. The response has `text` in place of `message`.

### Errors

Errors look like `{"error": {"status": 400, "message": "…"}}`.
- `400`: bad input
- `404`: unknown route
- `413`: body over 1 MB
- `500`: internal error (always answered as JSON)

## Safety guarantees (tested)

- **No control-token injection.** Text in `user` and `system` messages, including a literal `"<|finish|>"`, is always encoded as plain text.
- **No forged observations.** The model is blocked from emitting input-only tokens: `<|user|>`, `<|tool_result|>`, `<|tool_error|>`, `<|state|>`, `<|memory|>`, `<|context|>`, `<|tools|>` and `<|system|>`. It cannot fake a tool result or speak as the user.
- **No external calls.** All tokens come from the local Arouse model.

## Programmatic use

```python
import arouse
engine = arouse.load("artifacts/models/arouse-tiny")
print(engine.chat([arouse.Message("user", "hi")]).text)
arouse.generate("Every Monday", model="artifacts/models/arouse-tiny", max_new_tokens=20, temperature=0)
```
