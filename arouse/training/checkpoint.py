"""Crash-safe training checkpoints.

    <out_dir>/step_0000250/   model export (loadable by `arouse serve`) + trainer_state.pt
    <out_dir>/LATEST          name of the newest complete checkpoint

A checkpoint is written to `<name>.tmp` and renamed only when complete, then LATEST is
replaced atomically. A crash mid-save leaves the previous checkpoint intact.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch

from arouse.model.io import save_pretrained
from arouse.model.transformer import ArouseTransformer
from arouse.tokenizer import ArouseTokenizer

LATEST = "LATEST"
STATE_FILE = "trainer_state.pt"


def checkpoint_name(step: int) -> str:
    return f"step_{step:07d}"


def save_checkpoint(
    out_dir: Path,
    step: int,
    model: ArouseTransformer,
    tokenizer: ArouseTokenizer,
    state: dict[str, Any],
    keep: int,
) -> Path:
    final = out_dir / checkpoint_name(step)
    tmp = out_dir / (final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    save_pretrained(tmp, model, tokenizer, trained=True, train_steps=step, notes=f"checkpoint at step {step}")
    torch.save(state, tmp / STATE_FILE)
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    latest_tmp = out_dir / (LATEST + ".tmp")
    latest_tmp.write_text(final.name + "\n", encoding="utf-8")
    os.replace(latest_tmp, out_dir / LATEST)
    _rotate(out_dir, keep, final.name)
    return final


def _rotate(out_dir: Path, keep: int, protect: str) -> None:
    ckpts = sorted(p for p in out_dir.glob("step_*") if p.is_dir() and not p.name.endswith(".tmp"))
    for old in ckpts[:-keep]:
        if old.name != protect:
            shutil.rmtree(old, ignore_errors=True)


def latest_checkpoint(out_dir: Path) -> Path | None:
    pointer = out_dir / LATEST
    if not pointer.exists():
        return None
    path = out_dir / pointer.read_text(encoding="utf-8").strip()
    return path if (path / STATE_FILE).exists() else None


def load_trainer_state(ckpt: Path) -> dict[str, Any]:
    return torch.load(ckpt / STATE_FILE, map_location="cpu", weights_only=True)
