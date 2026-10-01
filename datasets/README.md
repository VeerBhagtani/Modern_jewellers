# Datasets

Rule: every file used for training must be listed in a `MANIFEST.json`, with its origin, license, and SHA-256. No indiscriminate scraping, and no private or copyrighted data without rights. Tests check that manifests match the files on disk.

| Dataset | Purpose | Origin |
|---|---|---|
| `tokenizer_tiny/` | Tokenizer development and tests only | Hand-written for Arouse, plus synthetic data from `scripts/build_tiny_tokenizer_corpus.py --seed 1234` (byte-reproducible) |
| `agent_v1/`, `agent_v2/` | Agent SFT data for `arouse-agent-s` (80k episodes each, about 30M tokens; v2 adds the plain-text calendar and explicit date lookup) plus AgentBench v1 | Synthetic, from `arouse/agent/synth.py` via `scripts/build_agent_data.py`. The large training file is regenerated on demand (reproducible by seed); `MANIFEST.json` records its SHA-256 |

`tokenizer_tiny` contents:
- `general.txt`, `code.txt`, `structured.txt`, `multilingual.txt`: original hand-written text
- `scheduling.jsonl`: 1,000 natural-language scheduling requests
- `agent_trajectories.jsonl`: 300 rendered episodes covering one-time, weekly, monthly, every-N-days and relative reminders, ambiguous requests that must be asked about, tool errors with retry or fail, and a missing file

Regenerate the agent data:

```bash
python scripts/build_agent_data.py --name agent_v2   # -> artifacts/datasets/agent_v2/train.jsonl + benchmarks/agentbench_v1/*.jsonl
```
