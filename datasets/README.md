# Datasets

Rule: every file used for training must be listed in a `MANIFEST.json`, with its origin, license, and SHA-256. No indiscriminate scraping, and no private or copyrighted data without rights. Tests check that manifests match the files on disk.

| Dataset | Purpose | Origin |
|---|---|---|
| `tokenizer_tiny/` | Tokenizer development and tests only | Hand-written for Arouse, plus synthetic data from `scripts/build_tiny_tokenizer_corpus.py --seed 1234` (byte-reproducible) |
| `agent_v1/`, `agent_v2/`, `agent_v3/` | Agent SFT data for `arouse-agent-s` plus AgentBench v1:<br>• v1 and v2: 80k episodes each, about 30M tokens. v2 adds the plain-text calendar and explicit date lookup<br>• v3: 100k episodes. Adds train-only phrasing variety for the measured weak spots: notes, deletes, one-time dates, out-of-scope questions, small talk, polite prefixes and suffixes. The held-out test split is unchanged, byte for byte | Synthetic, from `arouse/agent/synth.py` via `scripts/build_agent_data.py`. The large training file is regenerated on demand (reproducible by seed); `MANIFEST.json` records its SHA-256 |

`tokenizer_tiny` contents:
- `general.txt`, `code.txt`, `structured.txt`, `multilingual.txt`: original hand-written text
- `scheduling.jsonl`: 1,000 natural-language scheduling requests
- `agent_trajectories.jsonl`: 300 rendered episodes covering one-time, weekly, monthly, every-N-days and relative reminders, ambiguous requests that must be asked about, tool errors with retry or fail, and a missing file

Regenerate the agent data:

```bash
python scripts/build_agent_data.py --name agent_v3   # -> artifacts/datasets/agent_v3/train.jsonl (add --bench to rewrite the benchmark files)
```
