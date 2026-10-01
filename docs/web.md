# Arouse in the browser (`web/`)

The website runs the trained Arouse model entirely in the visitor's browser. There is no server-side model and no external AI service: the page downloads the weights once (11 MB) and runs every token on the visitor's CPU in a Web Worker.

```bash
python scripts/export_web.py --model models/arouse-agent-s --out web/model   # after training a new model
python -m http.server 8080 -d web                                            # open http://localhost:8080
```

Opening `web/index.html` directly from disk (`file://`) does not work, because browsers block Web Workers and `fetch` there. Use any static web server. `.github/workflows/pages.yml` publishes `web/` to GitHub Pages (enable it once under Settings → Pages → Source: GitHub Actions).

## Files

| File | What it is |
|---|---|
| `web/arouse.js` | JavaScript port of the Arouse runtime: byte-level BPE tokenizer, the Transformer (RoPE, GQA, SwiGLU, RMSNorm, KV cache, blocked prompt prefill), sampling, action protocol codec, copy- and date-constrained decoding, grounding and answer guards, the agent loop, the sandbox tools |
| `web/worker.js` | Loads the model and runs the agent off the main thread |
| `web/index.html` | Chat UI plus a workspace panel (reminders, notes, files) |
| `web/model/` | `config.json` (architecture and tensor table), `weights.bin` (float16), `tokenizer.json`, `response_vocab.txt`. Hosts that only serve text can use base64 weights instead: set `"weights_file": "weights.b64.txt", "weights_encoding": "base64"` in `config.json` |

## Behaviour

- **Workspace:** reminders, notes and files are stored in the browser's `localStorage`. Three sample files are included and marked as samples. Visitors can add their own text or CSV files (up to 200 KB each).
- **Firing reminders:** reminders fire while the page is open. The page checks every 15 seconds and shows a notice in the chat. One-time reminders are then removed; repeating ones move to their next run.
- **Steps:** each answer can be expanded to show the tool calls Arouse made and their results.
- **KV cache:** the KV cache is reused for the shared prompt prefix. The context includes the current minute, so most turns recompute the prompt. At this model size that takes a few seconds on a laptop.

## Parity with Python

`tests/test_web.py` runs `web/arouse.js` in Node (through `tests/web_harness.js`) and compares it with the Python implementation, using the website's own float16 weights on both sides:

- token ids for mixed scripts, emoji, JSON and benchmark messages: identical
- prompt token ids for benchmark episodes: identical
- next-token logits: equal within float rounding, with the same argmax
- copy-constraint continuations and runtime fallback answers: identical
- agent decisions (raw and system mode) and end-to-end runs on AgentBench cases:
  - identical whenever no resampling was needed
  - resamples use a different random generator (mulberry32 in JS, torch in Python), so those runs may differ

The test also checks that `web/model` is the released `models/arouse-agent-s`, rounded to float16.

## Limits

The model is the same 5.5M-parameter `arouse-agent-s` described in the README, with the same measured strengths and weaknesses. It only handles reminders, notes and files, and it can misread unusual phrasing.
