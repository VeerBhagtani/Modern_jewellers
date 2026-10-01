"""Arouse's built-in knowledge base and its search (the `kb.search` tool).

The model does not memorise facts: it routes a question to `kb.search`, and this
deterministic search returns a hand-written, source-checked answer (or "not found").
The model then answers with that text, and the runtime checks it against the tool
result. Unknown questions get an honest "I don't know that yet".

Scoring (mirrored exactly in web/arouse.js):
  terms    lowercase letter runs and digit runs ("22k" -> "22", "k"), minus single letters and
           stopwords, light suffix stemming, synonyms; "e-invoice"/"e-way" are one word
  idf      log(1 + N / df) over entries
  phrasing an entry's question or one of its alternative phrasings; strong = all phrasing terms,
           weak = answer terms
  known    query terms some entry contains; if at least as many query terms are unknown, the
           question is off-topic (not found)
  cover    sum over known terms of idf * (1 strong, 0.5 weak, 0 absent) / known weight
  prec     best phrasing: sum of idf of shared terms / sum of idf of that phrasing's terms
  score    cover + 0.25 * prec; found if cover >= 0.5 and at least one strong term is shared
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

PATH = Path(__file__).with_name("knowledge.json")
_WORD = re.compile(r"[a-z]+|[0-9]+")
_COMPOUND = re.compile(r"\be[- ](?=invoic|way)")  # "e-invoice" -> "einvoice", "e-way" -> "eway"
STOPWORDS = frozenset("""a an the is are was were be been being am do does did doing done have has had having i me my we our you your
it its this that these those what whats which who whom whose when where why how much many can could would should will shall
may might must of to in on at by for from with about as into than then so and or but if not no yes please tell explain
know want need like give show find get let us say says said mean means meaning there here any some all each every more
most other own same such only also just very too really kind sort way ways thing things something anything one
hi hey hello arouse ok okay thanks thank help today now current currently latest actually happen happens long list
""".split())
SYNONYMS = {"carat": "karat", "jewelry": "jewellery", "jeweller": "jewellery", "jewellers": "jewellery", "ornament": "jewellery",
            "ornaments": "jewellery", "rupee": "rs", "rupees": "rs", "inr": "rs", "lac": "lakh", "lacs": "lakh"}


NO_STEM = frozenset(["news", "gst", "gstin", "huid", "upi", "pan", "snf", "itc", "qrmp", "hsn", "igst", "cgst", "sgst", "always"])


def stem(w: str) -> str:
    """Light, deterministic suffix stripping: rates/rate, calculated/calculate, making/make."""
    w = SYNONYMS.get(w, w)
    if w in NO_STEM or w.isdigit():
        return w
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 3 and w.endswith("xes"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def terms(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(_COMPOUND.sub("e", text.lower())):
        if (len(w) > 1 or w.isdigit()) and w not in STOPWORDS:
            t = stem(w)
            if t not in out:
                out.append(t)
    return out


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    kb = json.loads(PATH.read_text(encoding="utf-8"))
    entries = kb["entries"]
    phrasings = [[set(terms(p)) for p in [e["q"], *e["alts"]]] for e in entries]
    strong = [set().union(*ps) for ps in phrasings]
    weak = [set(terms(e["a"])) for e in entries]
    df: dict[str, int] = {}
    for st, wk in zip(strong, weak, strict=True):
        for t in st | wk:
            df[t] = df.get(t, 0) + 1
    n = len(entries)
    idf = {t: math.log(1 + n / c) for t, c in df.items()}
    return {"kb": kb, "entries": entries, "phrasings": phrasings, "strong": strong, "weak": weak, "idf": idf, "n": n}


def search(query: str) -> dict[str, Any]:
    """Best entry for `query`: {"found": True, "id", "question", "answer"} or {"found": False}."""
    ix = load()
    idf, q = ix["idf"], terms(query)
    known = [t for t in q if t in idf]
    if not known or len(q) - len(known) >= len(known):  # mostly words the knowledge base has never seen
        return {"found": False}
    known_w = sum(idf[t] for t in known)
    best, best_score, best_cover = -1, 0.0, 0.0
    for i, (st, wk) in enumerate(zip(ix["strong"], ix["weak"], strict=True)):
        if not any(t in st for t in known):
            continue
        cover = sum(idf[t] * (1.0 if t in st else 0.5 if t in wk else 0.0) for t in known) / known_w
        prec = 0.0
        for ph in ix["phrasings"][i]:
            denom = sum(idf[t] for t in sorted(ph))  # sorted: same float sum as the JS port
            if denom:
                prec = max(prec, sum(idf[t] for t in known if t in ph) / denom)
        score = cover + 0.25 * prec
        if score > best_score + 1e-9:
            best, best_score, best_cover = i, score, cover
    if best < 0 or best_cover < 0.5:
        return {"found": False}
    e = ix["entries"][best]
    return {"found": True, "id": e["id"], "question": e["q"], "answer": e["a"]}
