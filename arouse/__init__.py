"""Arouse AI core: tokenizer, model, training, inference, agent protocol, evaluation.

Programmatic use (requires PyTorch):

    import arouse
    engine = arouse.load("artifacts/models/arouse-tiny")
    engine.chat([arouse.Message("user", "hi")])
    arouse.generate("Every Monday", model="artifacts/models/arouse-tiny", max_new_tokens=20)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0.dev0"

if TYPE_CHECKING:
    from arouse.inference import InferenceEngine

_ENGINES: dict[tuple[str, str], InferenceEngine] = {}


def load(model_dir: str, device: str = "cpu") -> InferenceEngine:
    """Load a saved model directory into an InferenceEngine (cached per path/device)."""
    from arouse.inference import InferenceEngine

    key = (str(model_dir), device)
    if key not in _ENGINES:
        _ENGINES[key] = InferenceEngine.from_pretrained(model_dir, device)
    return _ENGINES[key]


def generate(prompt: str, *, model: str, device: str = "cpu", **sampling: Any) -> str:
    """Raw text completion. `sampling`: max_new_tokens, temperature, top_k, top_p, seed."""
    from arouse.inference import SamplingParams

    return load(model, device).generate(prompt, SamplingParams(**sampling)).text


def __getattr__(name: str) -> Any:
    if name == "Message":  # lazy: keeps `import arouse` torch-free
        from arouse.inference import Message

        return Message
    raise AttributeError(name)
