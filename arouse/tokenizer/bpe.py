"""Byte-level BPE core: training and merge application. Pure Python, deterministic.

Token ID layout:
    [0, byte_offset)              special tokens
    [byte_offset, byte_offset+256) raw bytes 0x00..0xFF
    [byte_offset+256, ...)        merges, in rank order
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Mapping

Pair = tuple[int, int]


def train_bpe(
    chunk_counts: Mapping[bytes, int],
    num_merges: int,
    *,
    byte_offset: int,
    min_frequency: int = 2,
) -> list[Pair]:
    """Learn up to `num_merges` merges from chunk frequencies.

    Incremental algorithm: pair counts are updated only for chunks touched by a
    merge; a lazy max-heap finds the best pair. Ties break on the smallest pair
    IDs, so the result is fully deterministic.
    """
    words: list[list[int]] = []
    freqs: list[int] = []
    for chunk in sorted(chunk_counts):  # sorted -> order-independent result
        words.append([b + byte_offset for b in chunk])
        freqs.append(chunk_counts[chunk])

    pair_counts: dict[Pair, int] = defaultdict(int)
    where: dict[Pair, set[int]] = defaultdict(set)
    for wi, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freqs[wi]
            where[pair].add(wi)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[Pair] = []
    next_id = byte_offset + 256
    while len(merges) < num_merges and heap:
        neg, pair = heapq.heappop(heap)
        count = pair_counts.get(pair, 0)
        if count != -neg:  # stale heap entry
            if count > 0:
                heapq.heappush(heap, (-count, pair))
            continue
        if count < min_frequency:
            break

        merges.append(pair)
        new_id = next_id
        next_id += 1
        a, b = pair
        new_pairs: set[Pair] = set()
        for wi in where.pop(pair, ()):
            word, f = words[wi], freqs[wi]
            for p in zip(word, word[1:]):
                c = pair_counts[p] - f
                if c:
                    pair_counts[p] = c
                else:
                    del pair_counts[p]
            merged = _merge(word, a, b, new_id)
            words[wi] = merged
            for p in zip(merged, merged[1:]):
                pair_counts[p] += f
                where[p].add(wi)
                if new_id in p:
                    new_pairs.add(p)
        # Only pairs containing new_id can have increased; others are handled lazily.
        for p in new_pairs:
            heapq.heappush(heap, (-pair_counts[p], p))
    return merges


def _merge(ids: list[int], a: int, b: int, new_id: int) -> list[int]:
    out: list[int] = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def apply_merges(ids: list[int], ranks: Mapping[Pair, int], byte_offset: int) -> list[int]:
    """Repeatedly merge the lowest-rank pair present (same order as training)."""
    first_merge_id = byte_offset + 256
    while len(ids) >= 2:
        best_rank, best = None, None
        for pair in zip(ids, ids[1:]):
            r = ranks.get(pair)
            if r is not None and (best_rank is None or r < best_rank):
                best_rank, best = r, pair
        if best is None:
            break
        ids = _merge(ids, best[0], best[1], first_merge_id + best_rank)
    return ids
