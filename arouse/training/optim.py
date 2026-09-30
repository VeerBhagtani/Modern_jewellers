"""Optimizer and learning-rate schedule."""

from __future__ import annotations

import math

import torch
from torch import nn

from arouse.training.config import TrainingConfig


def lr_at(step: int, cfg: TrainingConfig) -> float:
    """Linear warmup to `lr`, then cosine decay to `min_lr` at `max_steps`. `step` is 0-based."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min((step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1), 1.0)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.AdamW:
    """AdamW; weight decay on matrices (incl. embeddings) only, never on norms/biases."""
    decay, no_decay = [], []
    for p in model.parameters():  # shared (tied) parameters appear once
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8)
