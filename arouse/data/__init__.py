"""Dataset pipeline, independent of the model: provenance, cleaning, dedup, split, packing, mixing."""

from arouse.data.config import DataConfig, DataSourceConfig

__all__ = ["DataConfig", "DataSourceConfig"]
