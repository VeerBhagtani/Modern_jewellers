# Data Pipeline and Training

```bash
arouse data prepare --config configs/data_tiny.yaml     # sources -> packed token streams
arouse train --config configs/train_tiny.yaml           # auto-resumes from the latest checkpoint
arouse serve --model checkpoints/arouse-tiny            # serves the latest checkpoint
```

## 1. Data (`arouse/data/`, independent of the model)

Every source in a data config must declare where it came from:

```yaml
- name: agent
  paths: [datasets/tokenizer_tiny/agent_trajectories.jsonl]
  format: jsonl              # or txt (split: file | paragraphs | lines)
  category: agent            # general | code | structured | instructions | agent | tool_use | scheduling | reasoning
  origin: scripts/build_tiny_tokenizer_corpus.py --seed 1234   # required
  license: project-owned                                        # required
  weight: 3.0                # mixing weight
  allow_special: true        # only for project-rendered data
  loss_on: all               # or "assistant": train only on Arouse's turns
```

`prepare` steps, in order:

| Step | What happens |
|---|---|
| Clean | Unicode NFC; `\r\n` → `\n`; strip control characters and trailing spaces; drop documents that are too short or more than 10% garbage |
| Dedup | Exact match, ignoring case and whitespace, **across all sources** (the first source listed keeps the copy) |
| Split | Train/val chosen by a hash of the content. Stable on every run and machine, and duplicates can never land on both sides |
| Tokenize | Each document is wrapped in `<|bos|> … <|eos|>`. Special tokens are recognized only for `allow_special` sources; any other text that looks like a special token is encoded as plain bytes |
| Mask | One byte per token: `1` = this token is a training target |
| Pack | `<source>.<split>.tokens.bin` (uint16) and `.mask.bin`, plus `meta.json` |

`meta.json` records each source's origin, license, file SHA-256s, document and token counts, the tokenizer fingerprint, and a **data fingerprint**.

## 2. Batching (`MixtureLoader`)

- Each batch row picks a source with probability proportional to its `weight`, then a random window of `seq_len + 1` tokens.
- Inputs are `window[:-1]`. Targets are `window[1:]`, with masked positions set to `-100`.
- Sources shorter than one window are **tiled** (repeated).
- **Batches depend only on `(seed, step, micro)`**, so resuming never replays or skips data.
- Validation uses fixed, non-overlapping windows per source, so evaluations are directly comparable over time.

## 3. Training (`arouse/training/`)

| Feature | Implementation |
|---|---|
| Optimizer | AdamW. Weight decay on matrices only; none on norms or biases |
| LR schedule | Linear warmup, then cosine decay to `min_lr` |
| Gradient accumulation | `grad_accum_steps` micro-batches per optimizer step |
| Mixed precision | `precision: bf16 \| fp16 \| fp32`. fp16 uses a gradient scaler on CUDA |
| Clipping | Global gradient norm (`grad_clip`) |
| Checkpoints | Every `checkpoint_interval` steps. Written to `.tmp`, then atomically renamed; `LATEST` is updated last; only the newest `keep_checkpoints` are kept |
| Resume | Automatic from `LATEST`. Restores weights, optimizer, scaler and RNG. Refuses to resume if the model config or data fingerprint changed |
| Metrics | `metrics.jsonl` (loss, learning rate, gradient norm, tokens/s, per-source validation loss) and console logs |
| Reproducibility | Configs and `run_info.json` (git commit, torch version, fingerprints) are saved at the start of a run |

Each checkpoint is also a complete model export (config, weights, tokenizer, meta), so `arouse serve` and `arouse.load()` can use it directly.

**Tested guarantee:** a run that crashes at step 6 and resumes produces **bitwise-identical** weights and losses at step 12 to an uninterrupted run (CPU, `tests/test_training.py`).

## Hardware

- CPU works for `arouse-tiny`: about 17k tokens/s on 4 cores.
- The 110M v0.1 model needs a CUDA GPU. Set `device: cuda` and `precision: bf16`. The code path is the same.
