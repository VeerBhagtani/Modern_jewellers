# Arouse Action Protocol (`arouse-action/1`)

Arouse never executes anything itself. On each turn it emits **exactly one** structured action. The runtime (or MDA) validates the action, executes it, and reports back the observation.

## Actions

```json
{"type": "tool_call", "tool": "scheduler.create", "arguments": {"task": "Study Geography", "date": "2026-10-02", "time": "08:00"}}
{"type": "ask_user",  "question": "What time should I remind you?"}
{"type": "finish",    "result": "Reminder set: Study Geography on 2026-10-02 at 08:00."}
{"type": "fail",      "error": "The scheduler is unavailable, so the reminder was not created."}
```

Validation rules (`arouse/protocol/actions.py`):
- `type` must be one of the four above.
- Every required field must be present, and no extra fields are allowed.
- Text fields must be non-empty strings.
- Tool names have a namespace (`^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$`).
- `arguments` must be an object, and is checked against the tool's schema (`arouse/protocol/tools.py`).

`tool_call` is the only non-terminal action. `ask_user`, `finish` and `fail` return control to the user.

## Model-side encoding

```
<|arouse|> [<|plan|> text] [<|verify|> text] <|ACTION|>{canonical JSON body without "type"}<|end|>
```

- **One token per action type.** The decision between calling a tool, asking, finishing or failing is a single prediction.
- **Canonical JSON:** compact, UTF-8, nested keys sorted. `"tool"` always comes first, so the model picks the tool before writing its arguments.
- **Piecewise encoding:** structure uses special-token IDs, and every text span is encoded with `allow_special=False`. A user or tool writing `<|finish|>` cannot inject a control token (tested).
- **Decoding** (`decode_turn`) works on token IDs and rejects anything that isn't `[plan][verify] ACTION body`.

## Episode (conversation) format

```
<|bos|><|system|>…<|end|><|context|>{now, weekday, tomorrow, next:{mon…sun}}<|end|><|tools|>[…]<|end|>
[<|memory|>…<|end|>] [<|state|>{…}<|end|>]
<|user|>…<|end|>
<|arouse|>…<|end|>  <|tool_result|>{…}<|end|>   (or <|tool_error|>{…}<|end|>)
<|arouse|>…<|end|> …
```

As JSON events (the `/v1/agent` API):

```json
{"type": "user", "content": "..."}
{"type": "arouse", "turn": {"plan": "...", "verify": "...", "action": {...}}}
{"type": "tool_result", "content": {"success": true, ...}}
{"type": "tool_error",  "content": {"success": false, "error": "...", "retryable": false}}
```

Training data and runtime prompts are produced by the **same** encoder (`arouse/agent/episode.py`), so there is no mismatch between what the model trained on and what it sees when serving.

## Built-in tools (v1)

| Tool | Arguments | Notes |
|---|---|---|
| `scheduler.create` | `task` plus one of: `date`+`time` · `time` only (next occurrence) · `in_minutes` · `time`+`repeat` | `repeat` = `{freq: daily\|weekly\|monthly, interval?, by_day?: [MO..SU], by_month_day?: [1..28]}` |
| `scheduler.list` | — | sorted by next run |
| `scheduler.delete` | `task_id` | |
| `notes.create` | `text` | |
| `file.read` | `path` | returns line count and a preview |
| `file.list` | — | |

**Design rule: the model understands, deterministic code computes.**
- The runtime supplies a calendar in `<|context|>`: tomorrow's date and the next date of each weekday.
- Relative and recurring requests are sent as offsets and rules. The scheduler computes the actual times.
- A 5M-parameter model can then be reliable, because it never has to do calendar arithmetic.

## Versioning

`PROTOCOL_VERSION = "arouse-action/1"` is returned by every agent endpoint.
- New *optional* fields and new tools are compatible changes.
- Anything else bumps the version.
- New special tokens take reserved slots (IDs 19–63), so existing token IDs never move.
