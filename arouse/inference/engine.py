"""Local inference engine: KV-cached autoregressive generation with streaming.

No network calls, no external model: logits come only from an ArouseTransformer.

Safety constraint: the model can never emit tokens that only the user/runtime may
write (<|user|>, <|tool_result|>, <|tool_error|>, <|state|>, ...). Arouse therefore
cannot forge a tool result or impersonate the user.
"""

from __future__ import annotations

import codecs
import dataclasses
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import torch

from arouse.inference.chat import Message, encode_chat
from arouse.inference.sampling import SamplingParams, sample_next
from arouse.model.io import check_compatible, load_pretrained
from arouse.model.transformer import ArouseTransformer
from arouse.tokenizer import INPUT_SEGMENTS, NUM_SPECIAL_SLOTS, ArouseTokenizer, Special

DEFAULT_STOP = (Special.END, Special.EOS)


@dataclasses.dataclass
class StreamEvent:
    token_id: int | None  # None only when nothing could be generated
    text: str  # decoded text delta ("" while a multi-byte char is incomplete)
    finish_reason: str | None = None  # set on the last event: "stop" | "length" | "context"


@dataclasses.dataclass
class Generation:
    text: str
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int


def banned_token_ids(tokenizer: ArouseTokenizer, model_vocab: int) -> list[int]:
    named = {int(t) for t in Special}
    banned = {int(Special.PAD), int(Special.BOS), int(Special.AROUSE), *map(int, INPUT_SEGMENTS)}
    banned |= {i for i in range(NUM_SPECIAL_SLOTS) if i not in named}  # reserved slots
    banned |= set(range(tokenizer.vocab_size, model_vocab))  # padded rows with no token
    return sorted(banned)


class InferenceEngine:
    def __init__(self, model: ArouseTransformer, tokenizer: ArouseTokenizer, meta: dict | None = None) -> None:
        check_compatible(model.config, tokenizer)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.meta = meta or {}
        self.device = next(model.parameters()).device
        self._banned = torch.tensor(banned_token_ids(tokenizer, model.config.vocab_size), dtype=torch.long)

    @classmethod
    def from_pretrained(cls, directory: str | Path, device: str = "cpu") -> InferenceEngine:
        model, tok, meta = load_pretrained(directory, device=device)
        return cls(model, tok, meta)

    @property
    def context_length(self) -> int:
        return self.model.config.context_length

    # --- core loop -----------------------------------------------------

    @torch.inference_mode()
    def stream(
        self,
        prompt_ids: Sequence[int],
        params: SamplingParams | None = None,
        stop_ids: Iterable[int] = DEFAULT_STOP,
        banned_ids: Iterable[int] = (),
    ) -> Iterator[StreamEvent]:
        params = params or SamplingParams()
        stop = {int(s) for s in stop_ids}
        banned = self._banned
        extra = [int(b) for b in banned_ids]
        if extra:
            banned = torch.cat((banned, torch.tensor(extra, dtype=torch.long)))
        if not prompt_ids:
            raise ValueError("prompt must contain at least one token")
        if len(prompt_ids) >= self.context_length:
            yield StreamEvent(None, "", "context")
            return

        gen = torch.Generator(device="cpu")
        if params.seed is not None:
            gen.manual_seed(params.seed)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        cache = self.model.new_kv_cache()
        ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.device)
        logits = self.model(ids, kv_cache=cache).logits[0, -1]
        pos = len(prompt_ids)

        for step in range(params.max_new_tokens):
            logits = logits.float().cpu()
            logits[banned] = float("-inf")
            tok = sample_next(logits, params, gen)
            if tok in stop:
                yield StreamEvent(tok, decoder.decode(b"", final=True), "stop")
                return
            if self.tokenizer.is_special(tok):
                text = decoder.decode(b"", final=True) + self.tokenizer.id_to_bytes(tok).decode()
            else:
                text = decoder.decode(self.tokenizer.id_to_bytes(tok))
            pos += 1
            if step == params.max_new_tokens - 1:
                yield StreamEvent(tok, text + decoder.decode(b"", final=True), "length")
                return
            if pos >= self.context_length:
                yield StreamEvent(tok, text + decoder.decode(b"", final=True), "context")
                return
            yield StreamEvent(tok, text)
            nxt = torch.tensor([[tok]], dtype=torch.long, device=self.device)
            logits = self.model(nxt, start_pos=pos - 1, kv_cache=cache).logits[0, -1]

    # --- conveniences ----------------------------------------------------

    def generate_ids(self, prompt_ids: Sequence[int], params: SamplingParams | None = None,
                     stop_ids: Iterable[int] = DEFAULT_STOP, banned_ids: Iterable[int] = ()) -> Generation:
        texts, ids, reason = [], [], "stop"
        for ev in self.stream(prompt_ids, params, stop_ids, banned_ids):
            texts.append(ev.text)
            if ev.token_id is not None and ev.finish_reason != "stop":
                ids.append(ev.token_id)
            reason = ev.finish_reason or reason
        return Generation("".join(texts), ids, reason, len(prompt_ids))

    def generate(self, prompt: str, params: SamplingParams | None = None, *, allow_special: bool = False) -> Generation:
        """Raw text completion."""
        return self.generate_ids(self.tokenizer.encode(prompt, allow_special=allow_special, add_bos=True), params)

    def chat_prompt(self, messages: Sequence[Message], params: SamplingParams) -> list[int]:
        budget = self.context_length - params.max_new_tokens
        return encode_chat(self.tokenizer, messages, max_tokens=max(budget, 1))

    def chat(self, messages: Sequence[Message], params: SamplingParams | None = None) -> Generation:
        params = params or SamplingParams()
        return self.generate_ids(self.chat_prompt(messages, params), params)

    def info(self) -> dict:
        cfg = self.model.config
        return {
            "name": cfg.name,
            "num_parameters": self.model.num_parameters(),
            "context_length": cfg.context_length,
            "vocab_size": self.tokenizer.vocab_size,
            "tokenizer_fingerprint": self.tokenizer.fingerprint(),
            "trained": bool(self.meta.get("trained", False)),
            "train_steps": int(self.meta.get("train_steps", 0)),
            "device": str(self.device),
        }
