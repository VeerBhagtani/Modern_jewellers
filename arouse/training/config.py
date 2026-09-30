"""Training run configuration."""

from __future__ import annotations

import dataclasses

from arouse.config import ConfigBase, ConfigError

PRECISIONS = ("fp32", "bf16", "fp16")


@dataclasses.dataclass
class TrainingConfig(ConfigBase):
    run_name: str
    model_config: str  # path to model YAML
    data_dir: str  # output of `arouse data prepare`
    out_dir: str  # checkpoints + metrics for this run
    seed: int = 1234
    init_from: str | None = None  # fine-tune: start from this model export / run dir (fresh optimizer)

    # batch: tokens per optimizer step = batch_size * grad_accum_steps * seq_len
    batch_size: int = 8  # micro-batch
    grad_accum_steps: int = 1
    seq_len: int | None = None  # None -> model context_length

    # optimisation
    max_steps: int = 1000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # 0 = off

    # system
    precision: str = "fp32"  # bf16 / fp16 autocast (fp16 uses a grad scaler on CUDA)
    device: str = "auto"  # auto | cpu | cuda | cuda:N
    threads: int = 0  # CPU threads; 0 = torch default

    # bookkeeping
    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int = 4
    checkpoint_interval: int = 250
    keep_checkpoints: int = 3

    def validate(self) -> None:
        for key in ("batch_size", "grad_accum_steps", "max_steps", "log_interval", "eval_interval",
                    "eval_batches", "checkpoint_interval", "keep_checkpoints"):
            if getattr(self, key) < 1:
                raise ConfigError(f"{key} must be >= 1")
        if self.seq_len is not None and self.seq_len < 2:
            raise ConfigError("seq_len must be >= 2")
        if not 0 < self.min_lr <= self.lr:
            raise ConfigError("need 0 < min_lr <= lr")
        if self.warmup_steps < 0 or self.warmup_steps >= self.max_steps:
            raise ConfigError("warmup_steps must be in [0, max_steps)")
        if self.weight_decay < 0 or self.grad_clip < 0 or self.threads < 0:
            raise ConfigError("weight_decay, grad_clip, threads must be >= 0")
        if not (0 < self.beta1 < 1 and 0 < self.beta2 < 1):
            raise ConfigError("betas must be in (0, 1)")
        if self.precision not in PRECISIONS:
            raise ConfigError(f"precision must be one of {PRECISIONS}")
        if not self.run_name.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise ConfigError("run_name must be alphanumeric/-/_/.")
