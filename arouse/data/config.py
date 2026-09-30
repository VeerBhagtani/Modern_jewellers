"""Dataset configuration: where every byte of training data comes from, and how much of it to use."""

from __future__ import annotations

import dataclasses

from arouse.config import ConfigBase, ConfigError

CATEGORIES = ("general", "code", "structured", "instructions", "agent", "tool_use", "scheduling", "reasoning")
FORMATS = ("txt", "jsonl")
SPLIT_MODES = ("file", "paragraphs", "lines")
LOSS_MODES = ("all", "assistant")
WINDOW_MODES = ("random", "doc_start")


@dataclasses.dataclass
class DataSourceConfig(ConfigBase):
    name: str
    paths: list[str]  # files or globs, relative to cwd
    category: str
    origin: str  # who wrote / produced it (required: no anonymous data)
    license: str  # usage rights (required)
    weight: float = 1.0  # relative probability of drawing a training sequence from this source
    format: str = "txt"
    text_field: str = "text"  # jsonl only
    split: str = "file"  # txt only: one doc per file / per blank-line paragraph / per line
    allow_special: bool = False  # True ONLY for project-rendered data whose special tokens are real structure
    loss_on: str = "all"  # "assistant": train only on tokens inside <|arouse|> ... <|end|> turns
    window: str = "random"  # "doc_start": training windows begin at a document start (whole episodes)

    def validate(self) -> None:
        if not self.name or not self.name.replace("_", "").replace("-", "").isalnum():
            raise ConfigError(f"source name must be alphanumeric/_/-: {self.name!r}")
        if not self.paths:
            raise ConfigError(f"source {self.name}: paths required")
        if not self.origin.strip() or not self.license.strip():
            raise ConfigError(f"source {self.name}: origin and license are required (provenance)")
        if self.category not in CATEGORIES:
            raise ConfigError(f"source {self.name}: category must be one of {CATEGORIES}")
        if self.format not in FORMATS:
            raise ConfigError(f"source {self.name}: format must be one of {FORMATS}")
        if self.split not in SPLIT_MODES:
            raise ConfigError(f"source {self.name}: split must be one of {SPLIT_MODES}")
        if self.loss_on not in LOSS_MODES:
            raise ConfigError(f"source {self.name}: loss_on must be one of {LOSS_MODES}")
        if self.window not in WINDOW_MODES:
            raise ConfigError(f"source {self.name}: window must be one of {WINDOW_MODES}")
        if self.weight <= 0:
            raise ConfigError(f"source {self.name}: weight must be > 0")


@dataclasses.dataclass
class DataConfig(ConfigBase):
    name: str
    tokenizer: str  # tokenizer directory
    output_dir: str
    sources: list[DataSourceConfig]
    val_fraction: float = 0.05
    min_chars: int = 1  # drop documents shorter than this after cleaning
    max_bad_char_ratio: float = 0.1  # drop documents with more control/replacement chars than this

    def validate(self) -> None:
        if not self.sources:
            raise ConfigError("at least one source is required")
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ConfigError(f"duplicate source names: {names}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ConfigError("val_fraction must be in [0, 1)")
        if self.min_chars < 1:
            raise ConfigError("min_chars must be >= 1")
        if not 0.0 <= self.max_bad_char_ratio <= 1.0:
            raise ConfigError("max_bad_char_ratio must be in [0, 1]")
