"""ArouseTokenizer: encode/decode, special tokens, save/load, fingerprint.

On disk (a directory):
    tokenizer.json  format, pattern, special tokens, merges checksum, training provenance
    merges.txt      one merge per line: "<left_id> <right_id>"  (rank = line order)
    vocab.json      human-readable id -> token (for inspection only; not loaded)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from arouse.tokenizer.bpe import Pair, apply_merges
from arouse.tokenizer.pretokenize import PATTERN_V1, compile_pattern
from arouse.tokenizer.special_tokens import (
    NUM_SPECIAL_SLOTS,
    SPECIAL_TOKENS_VERSION,
    Special,
    special_token_texts,
    split_on_special,
)

FORMAT = "arouse-byte-bpe"
FORMAT_VERSION = 1
BYTE_OFFSET = NUM_SPECIAL_SLOTS
FIRST_MERGE_ID = BYTE_OFFSET + 256
_MERGES_HEADER = "#arouse-merges v1"


class TokenizerError(ValueError):
    pass


def _merges_sha256(merges: Sequence[Pair]) -> str:
    text = "\n".join(f"{a} {b}" for a, b in merges)
    return hashlib.sha256(text.encode()).hexdigest()


class ArouseTokenizer:
    def __init__(
        self,
        merges: Sequence[Pair],
        *,
        pattern: str = PATTERN_V1,
        training_info: dict[str, Any] | None = None,
    ) -> None:
        self.pattern = pattern
        self.merges: list[Pair] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.training_info = training_info or {}
        self._regex = compile_pattern(pattern)

        self._special_texts = special_token_texts()
        self._special_to_id = {t: i for i, t in enumerate(self._special_texts)}

        # id -> bytes for non-special tokens
        self._bytes: list[bytes] = [b""] * BYTE_OFFSET + [bytes([i]) for i in range(256)]
        self._ranks: dict[Pair, int] = {}
        for rank, (a, b) in enumerate(self.merges):
            new_id = FIRST_MERGE_ID + rank
            if not (BYTE_OFFSET <= a < new_id and BYTE_OFFSET <= b < new_id):
                raise TokenizerError(f"merge {rank} ({a}, {b}) references invalid ids")
            if (a, b) in self._ranks:
                raise TokenizerError(f"duplicate merge ({a}, {b})")
            self._ranks[(a, b)] = rank
            self._bytes.append(self._bytes[a] + self._bytes[b])
        self._cache: dict[str, list[int]] = {}

    # --- vocabulary ----------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE_ID + len(self.merges)

    def is_special(self, token_id: int) -> bool:
        return 0 <= token_id < NUM_SPECIAL_SLOTS

    def token_to_id(self, special_text: str) -> int:
        if special_text not in self._special_to_id:
            raise KeyError(f"not a special token: {special_text!r}")
        return self._special_to_id[special_text]

    def id_to_bytes(self, token_id: int) -> bytes:
        """Raw bytes of a token (special tokens -> their text). Used for streaming decode."""
        if not 0 <= token_id < self.vocab_size:
            raise TokenizerError(f"token id {token_id} out of range [0, {self.vocab_size})")
        if self.is_special(token_id):
            return self._special_texts[token_id].encode()
        return self._bytes[token_id]

    # --- encode --------------------------------------------------------

    def encode(
        self,
        text: str,
        *,
        allow_special: bool = False,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Text -> ids.

        allow_special=False (default, use for ALL untrusted text): "<|finish|>" typed by a
        user is encoded as plain bytes, so users/tools cannot inject control tokens.
        allow_special=True: special-token strings map to their reserved ids.
        """
        ids: list[int] = [Special.BOS] if add_bos else []
        if allow_special:
            for piece, is_special in split_on_special(text):
                if is_special:
                    ids.append(self._special_to_id[piece])
                else:
                    ids.extend(self._encode_ordinary(piece))
        else:
            ids.extend(self._encode_ordinary(text))
        if add_eos:
            ids.append(Special.EOS)
        return ids

    def encode_batch(self, texts: Iterable[str], **kwargs: Any) -> list[list[int]]:
        return [self.encode(t, **kwargs) for t in texts]

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        cache = self._cache
        for chunk in self._regex.findall(text):
            ids = cache.get(chunk)
            if ids is None:
                ids = apply_merges([b + BYTE_OFFSET for b in chunk.encode("utf-8")], self._ranks, BYTE_OFFSET)
                if len(cache) >= 200_000:
                    cache.clear()
                cache[chunk] = ids
            out.extend(ids)
        return out

    # --- decode --------------------------------------------------------

    def decode(self, ids: Iterable[int], *, skip_special: bool = False) -> str:
        buf = bytearray()
        for i in ids:
            i = int(i)
            if skip_special and self.is_special(i):
                continue
            buf += self.id_to_bytes(i)
        return buf.decode("utf-8", errors="replace")

    # --- persistence ---------------------------------------------------

    def _functional_dict(self) -> dict[str, Any]:
        """Everything that affects encode/decode. Hashed by `fingerprint()`."""
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "pattern": self.pattern,
            "special_tokens_version": SPECIAL_TOKENS_VERSION,
            "special_tokens": self._special_texts,
            "byte_offset": BYTE_OFFSET,
            "vocab_size": self.vocab_size,
            "merges_sha256": _merges_sha256(self.merges),
        }

    def fingerprint(self) -> str:
        """Stable ID of this exact tokenizer. Checkpoints record it to prevent mismatches."""
        blob = json.dumps(self._functional_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        meta = {**self._functional_dict(), "fingerprint": self.fingerprint(), "training": self.training_info}
        (d / "tokenizer.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        lines = [_MERGES_HEADER] + [f"{a} {b}" for a, b in self.merges]
        (d / "merges.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        vocab = {
            str(i): (self._special_texts[i] if self.is_special(i) else self._bytes[i].decode("utf-8", "backslashreplace"))
            for i in range(self.vocab_size)
        }
        (d / "vocab.json").write_text(json.dumps(vocab, indent=0, ensure_ascii=False) + "\n", encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: str | Path) -> ArouseTokenizer:
        d = Path(directory)
        try:
            meta = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
            merge_lines = (d / "merges.txt").read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as e:
            raise TokenizerError(f"missing tokenizer file: {e.filename}") from e

        if meta.get("format") != FORMAT or meta.get("format_version") != FORMAT_VERSION:
            raise TokenizerError(f"unsupported tokenizer format: {meta.get('format')} v{meta.get('format_version')}")
        if meta.get("special_tokens") != special_token_texts():
            raise TokenizerError("special tokens in file do not match this code version")
        if not merge_lines or merge_lines[0] != _MERGES_HEADER:
            raise TokenizerError("merges.txt: bad header")
        try:
            merges = [tuple(int(x) for x in line.split()) for line in merge_lines[1:] if line]
        except ValueError as e:
            raise TokenizerError(f"merges.txt: {e}") from e
        if any(len(m) != 2 for m in merges):
            raise TokenizerError("merges.txt: each line must hold exactly two ids")
        if _merges_sha256(merges) != meta.get("merges_sha256"):
            raise TokenizerError("merges.txt checksum mismatch (corrupted or edited)")

        tok = cls(merges, pattern=meta["pattern"], training_info=meta.get("training", {}))  # type: ignore[arg-type]
        if tok.vocab_size != meta.get("vocab_size"):
            raise TokenizerError("vocab_size mismatch")
        if tok.fingerprint() != meta.get("fingerprint"):
            raise TokenizerError("fingerprint mismatch")
        return tok
