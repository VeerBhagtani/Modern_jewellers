"""Pretraining / fine-tuning loop.

Resumable: `Trainer(cfg).train()` continues from `<out_dir>/LATEST` if present. Because
batches depend only on (seed, step) and RNG state is checkpointed, a resumed run
reproduces the uninterrupted run exactly (verified by tests on CPU).
"""

from __future__ import annotations

import contextlib
import json
import math
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from arouse.config import ConfigError
from arouse.data.loader import Batch, MixtureLoader
from arouse.model.config import ModelConfig
from arouse.model.io import check_compatible, load_pretrained
from arouse.model.transformer import ArouseTransformer
from arouse.tokenizer import ArouseTokenizer
from arouse.training.checkpoint import latest_checkpoint, load_trainer_state, save_checkpoint
from arouse.training.config import TrainingConfig
from arouse.training.optim import build_optimizer, lr_at

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError("device 'cuda' requested but CUDA is not available")
    return dev


class Trainer:
    def __init__(self, cfg: TrainingConfig, *, log: Callable[[str], None] = print) -> None:
        self.cfg = cfg
        self.log = log
        self.out = Path(cfg.out_dir)
        self.device = resolve_device(cfg.device)
        if cfg.threads:
            torch.set_num_threads(cfg.threads)

        self.model_cfg = ModelConfig.from_yaml(cfg.model_config).resolved()
        self.seq_len = cfg.seq_len or self.model_cfg.context_length
        if self.seq_len > self.model_cfg.context_length:
            raise ConfigError("seq_len exceeds the model's context_length")
        self.data = MixtureLoader(cfg.data_dir, self.seq_len, seed=cfg.seed)
        tok_dir = Path(self.data.meta["config"]["tokenizer"])
        self.tokenizer = ArouseTokenizer.load(tok_dir)
        if self.tokenizer.fingerprint() != self.data.tokenizer_fingerprint:
            raise ConfigError("tokenizer does not match the one the data was prepared with")
        check_compatible(self.model_cfg, self.tokenizer)

        torch.manual_seed(cfg.seed)
        self.model = ArouseTransformer(self.model_cfg).to(self.device)
        self.optimizer = build_optimizer(self.model, cfg)
        self.amp_dtype = _DTYPES.get(cfg.precision)
        use_scaler = cfg.precision == "fp16" and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        self.step = 0  # number of completed optimizer steps
        self.metrics_path = self.out / "metrics.jsonl"

    # --- state -------------------------------------------------------------

    def _state(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "training_config": self.cfg.to_dict(),
            "model_config": self.model_cfg.to_dict(),
            "data_fingerprint": self.data.data_fingerprint,
        }

    def _resume(self) -> bool:
        ckpt = latest_checkpoint(self.out)
        if ckpt is None:
            return False
        state = load_trainer_state(ckpt)
        if state["model_config"] != self.model_cfg.to_dict():
            raise ConfigError(f"model config differs from checkpoint {ckpt}")
        if state["data_fingerprint"] != self.data.data_fingerprint:
            raise ConfigError(f"prepared data differs from the data used by checkpoint {ckpt}")
        weights = torch.load(ckpt / "model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(weights)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.step = state["step"]
        self.log(f"resumed from {ckpt.name} (step {self.step})")
        return True

    def _init_from(self) -> None:
        model, tok, meta = load_pretrained(self.cfg.init_from, device=self.device)
        old, new = model.config.to_dict(), self.model_cfg.to_dict()
        # The context length is not part of the weights (RoPE positions are computed, not learned),
        # so a model may be fine-tuned at a longer context.
        if {**old, "context_length": 0} != {**new, "context_length": 0}:
            raise ConfigError("init_from model has a different architecture than model_config")
        if old["context_length"] != new["context_length"]:
            self.log(f"context length {old['context_length']} -> {new['context_length']} (weights unchanged)")
        if tok.fingerprint() != self.tokenizer.fingerprint():
            raise ConfigError("init_from model uses a different tokenizer than the prepared data")
        self.model.load_state_dict(model.state_dict())
        self.log(f"initialised from {self.cfg.init_from} (trained {meta.get('train_steps', 0)} steps)")

    def save(self) -> Path:
        path = save_checkpoint(self.out, self.step, self.model, self.tokenizer, self._state(), self.cfg.keep_checkpoints)
        self.log(f"saved {path}")
        return path

    # --- loop --------------------------------------------------------------

    def _autocast(self) -> contextlib.AbstractContextManager:
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.amp_dtype)

    def _to_device(self, b: Batch) -> Batch:
        return Batch(b.inputs.to(self.device, non_blocking=True), b.targets.to(self.device, non_blocking=True))

    def train_step(self) -> dict[str, float]:
        cfg = self.cfg
        lr = lr_at(self.step, cfg)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss, tokens = 0.0, 0
        for micro in range(cfg.grad_accum_steps):
            b = self._to_device(self.data.train_batch(self.step, cfg.batch_size, micro))
            with self._autocast():
                loss = self.model(b.inputs, targets=b.targets).loss
            self.scaler.scale(loss / cfg.grad_accum_steps).backward()
            total_loss += loss.item() / cfg.grad_accum_steps
            tokens += b.target_tokens
        self.scaler.unscale_(self.optimizer)
        if cfg.grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip))
        else:
            grad_norm = float(torch.norm(torch.stack([p.grad.norm() for p in self.model.parameters() if p.grad is not None])))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.step += 1
        return {"loss": total_loss, "lr": lr, "grad_norm": grad_norm, "tokens": tokens}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        out: dict[str, float] = {}
        weighted, wsum = 0.0, 0.0
        weights = {s["name"]: s["weight"] for s in self.data.meta["sources"]}
        for name, batches in self.data.val_batches(self.cfg.batch_size, self.cfg.eval_batches).items():
            losses, counts = [], []
            for b in batches:
                b = self._to_device(b)
                with self._autocast():
                    losses.append(self.model(b.inputs, targets=b.targets).loss.item())
                counts.append(b.target_tokens)
            if sum(counts):
                val = sum(loss * c for loss, c in zip(losses, counts)) / sum(counts)
                out[f"val/{name}"] = val
                weighted += weights[name] * val
                wsum += weights[name]
        if wsum:
            out["val/mixture"] = weighted / wsum
        return out

    def _write_metrics(self, row: dict[str, Any]) -> None:
        with open(self.metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def train(self) -> dict[str, Any]:
        cfg = self.cfg
        self.out.mkdir(parents=True, exist_ok=True)
        resumed = self._resume()
        if not resumed:
            if cfg.init_from:
                self._init_from()
            cfg.to_yaml(self.out / "training_config.yaml")
            self.model_cfg.to_yaml(self.out / "model_config.yaml")
            run_info = {
                "run_name": cfg.run_name,
                "git_commit": _git_commit(),
                "torch": torch.__version__,
                "device": str(self.device),
                "num_parameters": self.model.num_parameters(),
                "data_fingerprint": self.data.data_fingerprint,
                "tokenizer_fingerprint": self.tokenizer.fingerprint(),
                "tokens_per_step": cfg.batch_size * cfg.grad_accum_steps * self.seq_len,
                "init_from": cfg.init_from,
            }
            (self.out / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
            self.log(f"run {cfg.run_name}: {self.model_cfg.summary()} on {self.device}")

        last: dict[str, Any] = {}
        t0, tok_count = time.perf_counter(), 0
        while self.step < cfg.max_steps:
            stats = self.train_step()
            if not math.isfinite(stats["loss"]):
                raise RuntimeError(f"non-finite loss at step {self.step}; last checkpoint is intact")
            tok_count += stats["tokens"]
            row: dict[str, Any] = {"step": self.step, **stats}
            if self.step % cfg.log_interval == 0 or self.step == cfg.max_steps:
                dt = time.perf_counter() - t0
                row["tokens_per_sec"] = tok_count / dt if dt > 0 else 0.0
                self.log(f"step {self.step:>6} | loss {stats['loss']:.4f} | lr {stats['lr']:.2e} | "
                         f"gnorm {stats['grad_norm']:.2f} | {row['tokens_per_sec']:.0f} tok/s")
                t0, tok_count = time.perf_counter(), 0
            if self.step % cfg.eval_interval == 0 or self.step == cfg.max_steps:
                ev = self.evaluate()
                row.update(ev)
                if ev:
                    self.log("eval   | " + " | ".join(f"{k} {v:.4f}" for k, v in ev.items()))
            self._write_metrics(row)
            last = row
            if self.step % cfg.checkpoint_interval == 0 or self.step == cfg.max_steps:
                self.save()
        return last
