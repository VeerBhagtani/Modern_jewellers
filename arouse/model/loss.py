"""Next-token cross-entropy. `targets` are aligned with logits (already shifted by the data
pipeline); positions set to IGNORE_INDEX (e.g. prompt/user tokens) contribute nothing."""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    flat_logits = logits.float().reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    total = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, reduction="sum")
    count = (flat_targets != ignore_index).sum().clamp(min=1)  # all-masked batch -> 0, not NaN
    return total / count
