# Arouse Tokenizer (v1)

Byte-level BPE, implemented from scratch in pure Python (`arouse/tokenizer/`). No external tokenizer libraries.

## ID layout

| IDs | Content |
|---|---|
| `0–63` | Special tokens (19 named + 45 `<|reserved_N|>`) |
| `64–319` | Raw bytes `0x00–0xFF` |
| `320+` | Learned merges, in rank order |

- **Byte-level** → any UTF-8 text encodes. No unknown tokens, no lossy normalization.
- **Special IDs are frozen.** New tokens take a reserved slot, so byte/merge IDs never move and old checkpoints stay valid.

## Pre-tokenization (`PATTERN_V1`)

Text is first split into chunks. Merges never cross chunk boundaries.

| Rule | Example | Why |
|---|---|---|
| Letters + optional 1 leading space/punct | `' hello'`, `'"tool'` | Compact words and JSON keys |
| Letters include combining marks | `'सुबह'` stays whole | Hindi/Indic words don't shatter |
| `_` + letters | `task_id` → `task`, `_id` | snake_case tool args |
| **Single digits** | `2026` → `2`,`0`,`2`,`6` | Same digits always get the same tokens, which helps with dates, times, and arithmetic |
| Punctuation runs | `'":"'`, `'"}}'` | JSON structure compresses well |
| Whitespace / newlines | `'\n\n'`, `'    '` | Code indentation |
| Fallback `[\s\S]` | — | Never drops a character (tested by fuzzing) |

Only the standard `re` module is used. The combining-mark class is built from `unicodedata`. The final pattern string is saved **inside** each tokenizer file, so a trained tokenizer does not depend on the Python version that loads it.

## Special tokens

| ID | Token | Group | Meaning |
|---|---|---|---|
| 0 | `<|pad|>` | control | padding (never predicted) |
| 1 | `<|bos|>` | control | start of sequence |
| 2 | `<|eos|>` | control | end of document/episode |
| 3 | `<|end|>` | control | closes every segment/turn |
| 4 | `<|system|>` | input | operator instructions |
| 5 | `<|tools|>` | input | JSON list of available tools |
| 6 | `<|context|>` | input | runtime facts: `now`, `timezone` (needed to resolve "tomorrow") |
| 7 | `<|memory|>` | input | retrieved long-term memory |
| 8 | `<|state|>` | input | task-state JSON (resume, don't restart) |
| 9 | `<|user|>` | input | user message |
| 10 | `<|tool_result|>` | input | successful tool output |
| 11 | `<|tool_error|>` | input | **failed** tool output |
| 12 | `<|arouse|>` | model | start of Arouse's turn |
| 13 | `<|plan|>` | model | short plan |
| 14 | `<|verify|>` | model | check observations before acting/claiming done |
| 15 | `<|tool_call|>` | action | `{"tool":…,"arguments":{…}}` |
| 16 | `<|ask_user|>` | action | `{"question":…}` |
| 17 | `<|finish|>` | action | `{"result":…}` (verified complete) |
| 18 | `<|fail|>` | action | `{"error":…}` (cannot complete) |
| 19–63 | `<|reserved_N|>` | — | future use |

## Sequence grammar

```
<|bos|> segment* <|eos|>
segment     := ROLE content <|end|>
arouse turn := <|arouse|> [<|plan|> text] [<|verify|> text] (ACTION json | text) <|end|>
```

Example (one training episode):

```
<|bos|><|system|>You are Arouse…<|end|>
<|context|>{"now":"2026-09-30T09:15","timezone":"Asia/Kolkata"}<|end|>
<|tools|>[{"name":"scheduler.create",…}]<|end|>
<|user|>Every Monday at 9 AM remind me to check sales.<|end|>
<|arouse|><|plan|>Create a weekly recurring reminder.<|tool_call|>{"tool":"scheduler.create","arguments":{"task":"Check sales","date":"2026-10-05","time":"09:00","repeat":{"freq":"weekly","interval":1,"by_day":["MO"]}}}<|end|>
<|tool_result|>{"success":true,"task_id":"123"}<|end|>
<|arouse|><|verify|>scheduler.create succeeded with task_id 123.<|finish|>{"result":"The weekly reminder has been created."}<|end|><|eos|>
```

(Line breaks are for display only; real sequences have none between segments.)

## Design decisions (vs. the brief's example)

1. **The action type is one token** (`<|tool_call|>`, `<|ask_user|>`, `<|finish|>`, `<|fail|>`) rather than a `"type"` field inside the JSON.
   - The most important decision is a single prediction. Decoding can restrict it to allowed actions, and the benchmark can score it directly.
   - The mapping to protocol JSON is lossless: `{"type": X, **body}` ↔ `<|X|>body`. The versioned, schema-validated protocol arrives in Milestone 4.
2. **No closing tags inside a turn.** `<|plan|>` runs until the next special token, which saves tokens and removes one class of malformed output.
3. **`<|tool_error|>` is separate from `<|tool_result|>`.** Failure is a one-token signal the model cannot miss. This is aimed at stopping the model from claiming success after a failed tool.
4. **`<|verify|>` is a first-class token.** Training data puts a verification step before `<|finish|>`. Tests enforce that `<|finish|>` only ever follows a successful result (`tests/test_datasets.py`).
5. **`<|context|>`** carries `now` and the timezone. Without them, "tomorrow" and "in 2 hours" cannot be resolved without guessing.
6. **Injection-safe by default.** `encode(text)` treats `"<|finish|>"` typed by a user or returned by a tool as plain bytes. Only `encode(..., allow_special=True)`, used by the runtime for text it builds itself, produces control tokens.
7. **Compact canonical JSON** (`separators=(",", ":")`). One serialization means consistent tokens (`'":"'`, `'","'` become single tokens).

## Files on disk

```
tokenizer.json   format, version, pattern, 64 special tokens, vocab_size, merges_sha256,
                 fingerprint, training provenance (source paths + sha256)
merges.txt       "#arouse-merges v1" header, then "<left_id> <right_id>" per line
vocab.json       id -> readable token (inspection only)
```

`load()` verifies the format version, the special-token table, the merges checksum, the vocab size, and the fingerprint. Checkpoints will record the 16-hex-character `fingerprint()` so a model can't be paired with the wrong tokenizer.

## Performance (measured, 4-core CPU, single thread)

- Chunk counting: ~6 MB/s.
- `train_bpe`: 16k merges on 54k unique chunks in ~5 s. It uses incremental pair counts and a lazy max-heap, and is verified against a brute-force reference implementation on 30 random corpora.
- Tiny corpus (357 KB) → 1,508-token vocab in 0.3 s; 4.07 bytes/token across the whole corpus.

## Known limitations

- The tiny corpus cannot reach the 32,768-token target; the trainer stops early when no pair meets `min_frequency`. The production tokenizer needs the real corpus (Milestone 3).
- Pure-Python counting is fine up to about a few GB (`max_bytes` lets you sample). Parallel counting can be added if it becomes a bottleneck.
- Scripts without spaces (Chinese, Japanese) are split only by the letter rule. They work, but compress less well.
