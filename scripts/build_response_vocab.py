#!/usr/bin/env python3
"""Extract Arouse's response vocabulary: words used in its own answers (finish / ask_user / fail)
that do NOT come from the episode's content (user messages, tool results). These are the
template words it was trained to speak; everything else in an answer must be grounded.

Usage: python scripts/build_response_vocab.py  ->  arouse/agent/response_vocab.txt
"""

from __future__ import annotations

import json
from pathlib import Path

from arouse.agent.grounding import answer_words
from arouse.agent.synth import generate


def main() -> None:
    vocab: set[str] = set()
    for ep in generate(6000, seed=555) + generate(1000, seed=556, split="test"):
        content = set()
        for ev in ep["events"]:
            if ev["type"] == "user":
                content |= set(answer_words(ev["content"]))
            elif ev["type"] in ("tool_result", "tool_error"):
                content |= set(answer_words(json.dumps(ev["content"])))
        for ev in ep["events"]:
            if ev["type"] == "arouse":
                a = ev["turn"]["action"]
                text = a.get("result") or a.get("question") or a.get("error")
                if text:
                    vocab |= {w for w in answer_words(text) if w not in content and not any(c.isdigit() for c in w)}
    out = Path("arouse/agent/response_vocab.txt")
    out.write_text("\n".join(sorted(vocab)) + "\n", encoding="utf-8")
    print(f"{len(vocab)} words -> {out}")


if __name__ == "__main__":
    main()
