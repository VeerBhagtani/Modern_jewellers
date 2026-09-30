"""Arouse decoder-only Transformer.

Config is torch-free; importing the layers requires PyTorch.
"""

from arouse.model.config import PRESETS, ModelConfig, get_preset

__all__ = ["ModelConfig", "PRESETS", "get_preset"]
