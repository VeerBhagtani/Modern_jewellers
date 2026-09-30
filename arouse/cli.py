"""`arouse` command-line interface.

    arouse model info --config configs/model.yaml
    arouse model init --config configs/model_tiny.yaml --tokenizer DIR --out DIR   (random weights)
    arouse data prepare --config configs/data_tiny.yaml
    arouse train --config configs/train_tiny.yaml [--max-steps N]   (auto-resumes)
    arouse agent --model DIR                    terminal agent with sandbox tools
    arouse bench --model DIR --file benchmarks/agentbench_v1/test.jsonl [--out FILE]
    arouse chat  --model DIR                    terminal chat (streaming)
    arouse serve --model DIR [--port 8000]      local API + chat UI at http://127.0.0.1:8000
    arouse tokenizer train --config configs/tokenizer_tiny.yaml
    arouse tokenizer encode --tokenizer DIR [--allow-special] TEXT
    arouse tokenizer decode --tokenizer DIR ID [ID ...]
    arouse tokenizer stats --tokenizer DIR FILE [FILE ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from arouse.model.config import ModelConfig
from arouse.tokenizer import ArouseTokenizer, TokenizerTrainingConfig, train_tokenizer
from arouse.tokenizer.trainer import iter_documents

if TYPE_CHECKING:
    from arouse.inference import InferenceEngine, SamplingParams


def _model_info(a: argparse.Namespace) -> int:
    cfg = ModelConfig.from_yaml(a.config)
    print(cfg.summary())
    print(json.dumps(cfg.parameter_counts(), indent=2))
    return 0


def _model_init(a: argparse.Namespace) -> int:
    import torch

    from arouse.model.io import save_pretrained
    from arouse.model.transformer import ArouseTransformer

    cfg = ModelConfig.from_yaml(a.config)
    tok = ArouseTokenizer.load(a.tokenizer)
    torch.manual_seed(a.seed)
    model = ArouseTransformer(cfg)
    out = save_pretrained(a.out, model, tok, trained=False, notes=f"random init, seed {a.seed}")
    print(f"{cfg.summary()}\nsaved UNTRAINED model -> {out}")
    return 0


def _load_engine(model_dir: str) -> InferenceEngine:
    from arouse.inference import InferenceEngine

    return InferenceEngine.from_pretrained(model_dir)


def _sampling(a: argparse.Namespace) -> SamplingParams:
    from arouse.inference import SamplingParams

    return SamplingParams(max_new_tokens=a.max_tokens, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, seed=a.seed)


def _chat(a: argparse.Namespace) -> int:
    from arouse.inference import Message

    engine = _load_engine(a.model)
    info = engine.info()
    print(f"Arouse chat - {info['name']} ({'trained' if info['trained'] else 'UNTRAINED: random weights'}). Ctrl+D to exit.")
    history: list[Message] = [Message("system", a.system)] if a.system else []
    params = _sampling(a)
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        history.append(Message("user", text))
        print("arouse> ", end="", flush=True)
        reply = []
        for ev in engine.stream(engine.chat_prompt(history, params), params):
            print(ev.text, end="", flush=True)
            reply.append(ev.text)
        print()
        history.append(Message("assistant", "".join(reply)))


def _serve(a: argparse.Namespace) -> int:
    from arouse.api.server import make_server

    engine = _load_engine(a.model)
    srv = make_server(engine, a.host, a.port, verbose=a.verbose)
    info = engine.info()
    print(f"Arouse API: {info['name']} ({'trained' if info['trained'] else 'UNTRAINED'}) on http://{a.host}:{srv.server_port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def _data_prepare(a: argparse.Namespace) -> int:
    from arouse.data.config import DataConfig
    from arouse.data.prepare import prepare

    meta = prepare(DataConfig.from_yaml(a.config))
    for s in meta["sources"]:
        tr, va = s["splits"]["train"], s["splits"]["val"]
        print(f"{s['name']:>12} [{s['category']}] docs {s['docs']['train']}/{s['docs']['val']} (train/val) "
              f"tokens {tr['tokens']}/{va['tokens']} dropped {s['docs_dropped_clean']} dup {s['docs_dropped_dup']} "
              f"weight {s['weight']}")
    print(f"data fingerprint {meta['data_fingerprint']} -> {meta['config']['output_dir']}")
    return 0


def _train(a: argparse.Namespace) -> int:
    from arouse.training.config import TrainingConfig
    from arouse.training.trainer import Trainer

    cfg = TrainingConfig.from_yaml(a.config)
    if a.max_steps:
        cfg = cfg.replace(max_steps=a.max_steps)
    Trainer(cfg).train()
    return 0


def _agent(a: argparse.Namespace) -> int:
    from datetime import datetime

    from arouse.agent.context import build_context
    from arouse.agent.episode import Header
    from arouse.agent.runtime import AgentRuntime
    from arouse.agent.tools import REGISTRY, Sandbox
    from arouse.api.agent_service import DEMO_FILES

    engine = _load_engine(a.model)
    runtime = AgentRuntime(engine, REGISTRY)
    sandbox = Sandbox(datetime.now().replace(second=0, microsecond=0), DEMO_FILES)
    events: list[dict] = []
    print(f"Arouse agent - {engine.info()['name']}. Sandbox files: {', '.join(sorted(DEMO_FILES))}. Ctrl+D to exit.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        now = datetime.now().replace(second=0, microsecond=0)
        sandbox.now = now
        events.append({"type": "user", "content": text})
        res = runtime.run(Header(context=build_context(now, "local"), tools=REGISTRY.names()), events, sandbox.execute)
        events.extend(res.events)
        for ev in res.events:
            if ev["type"] == "arouse" and ev["turn"]["action"]["type"] == "tool_call":
                act = ev["turn"]["action"]
                print(f"  -> {act['tool']} {json.dumps(act['arguments'])}")
            elif ev["type"] in ("tool_result", "tool_error"):
                print(f"  <- {ev['type']}: {json.dumps(ev['content'])[:160]}")
        f = res.final
        print(f"arouse> {f.result or f.question or f.error}" + ("" if f.type != "fail" else "  [failed]"))


def _bench(a: argparse.Namespace) -> int:
    from arouse.evaluation.agentbench import run_benchmark, summary_lines

    report = run_benchmark(a.model, a.file, decision_limit=a.limit or None, e2e_limit=a.e2e_limit or None)
    for line in summary_lines(report):
        print(line)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report -> {a.out}")
    return 0


def _add_sampling_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)


def _tok_train(a: argparse.Namespace) -> int:
    cfg = TokenizerTrainingConfig.from_yaml(a.config)
    if a.output_dir:
        cfg = cfg.replace(output_dir=a.output_dir)
    t0 = time.perf_counter()
    tok = train_tokenizer(cfg)
    out = tok.save(cfg.output_dir)
    info = tok.training_info
    print(
        f"trained vocab={tok.vocab_size} (target {cfg.vocab_size}) merges={len(tok.merges)} "
        f"bytes={info['bytes_used']} chunks={info['unique_chunks']} "
        f"time={time.perf_counter() - t0:.1f}s fingerprint={tok.fingerprint()} -> {out}"
    )
    if tok.vocab_size < cfg.vocab_size:
        print("note: corpus too small to reach target vocab (min_frequency stop)")
    return 0


def _tok_encode(a: argparse.Namespace) -> int:
    tok = ArouseTokenizer.load(a.tokenizer)
    ids = tok.encode(a.text, allow_special=a.allow_special)
    print(json.dumps(ids))
    if a.verbose:
        for i in ids:
            print(f"{i:>6}  {tok.id_to_bytes(i)!r}")
    return 0


def _tok_decode(a: argparse.Namespace) -> int:
    tok = ArouseTokenizer.load(a.tokenizer)
    print(tok.decode(a.ids))
    return 0


def _tok_stats(a: argparse.Namespace) -> int:
    tok = ArouseTokenizer.load(a.tokenizer)
    n_bytes = n_tokens = 0
    for f in a.files:
        for doc in iter_documents(Path(f)):
            n_bytes += len(doc.encode("utf-8"))
            n_tokens += len(tok.encode(doc, allow_special=True))
    ratio = n_bytes / n_tokens if n_tokens else 0.0
    print(f"bytes={n_bytes} tokens={n_tokens} bytes/token={ratio:.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arouse", description="Arouse AI core tools")
    sub = p.add_subparsers(dest="group", required=True)

    model = sub.add_parser("model").add_subparsers(dest="cmd", required=True)
    info = model.add_parser("info", help="show architecture + parameter count")
    info.add_argument("--config", default="configs/model.yaml")
    info.set_defaults(fn=_model_info)
    init = model.add_parser("init", help="save a randomly initialised model (for pipeline/UI testing)")
    init.add_argument("--config", required=True)
    init.add_argument("--tokenizer", required=True)
    init.add_argument("--out", required=True)
    init.add_argument("--seed", type=int, default=0)
    init.set_defaults(fn=_model_init)

    data = sub.add_parser("data").add_subparsers(dest="cmd", required=True)
    prep = data.add_parser("prepare", help="clean, dedup, split, tokenize and pack a dataset")
    prep.add_argument("--config", required=True)
    prep.set_defaults(fn=_data_prepare)

    train = sub.add_parser("train", help="train a model (auto-resumes from the latest checkpoint)")
    train.add_argument("--config", required=True)
    train.add_argument("--max-steps", type=int, default=0)
    train.set_defaults(fn=_train)

    agent = sub.add_parser("agent", help="terminal agent (runs tools in a local sandbox)")
    agent.add_argument("--model", required=True, help="model directory")
    agent.set_defaults(fn=_agent)

    bench = sub.add_parser("bench", help="run Arouse AgentBench")
    bench.add_argument("--model", required=True)
    bench.add_argument("--file", default="benchmarks/agentbench_v1/test.jsonl")
    bench.add_argument("--out", default="")
    bench.add_argument("--limit", type=int, default=0, help="max decisions (0 = all)")
    bench.add_argument("--e2e-limit", type=int, default=0, help="max end-to-end episodes (0 = all)")
    bench.set_defaults(fn=_bench)

    chat = sub.add_parser("chat", help="interactive terminal chat")
    chat.add_argument("--model", required=True, help="model directory")
    chat.add_argument("--system", default="")
    _add_sampling_args(chat)
    chat.set_defaults(fn=_chat)

    serve = sub.add_parser("serve", help="local HTTP API + chat UI")
    serve.add_argument("--model", required=True, help="model directory")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("-v", "--verbose", action="store_true")
    serve.set_defaults(fn=_serve)

    tok = sub.add_parser("tokenizer").add_subparsers(dest="cmd", required=True)
    tr = tok.add_parser("train", help="train a tokenizer from a YAML config")
    tr.add_argument("--config", required=True)
    tr.add_argument("--output-dir")
    tr.set_defaults(fn=_tok_train)

    enc = tok.add_parser("encode")
    enc.add_argument("--tokenizer", required=True)
    enc.add_argument("--allow-special", action="store_true")
    enc.add_argument("-v", "--verbose", action="store_true")
    enc.add_argument("text")
    enc.set_defaults(fn=_tok_encode)

    dec = tok.add_parser("decode")
    dec.add_argument("--tokenizer", required=True)
    dec.add_argument("ids", nargs="+", type=int)
    dec.set_defaults(fn=_tok_decode)

    st = tok.add_parser("stats", help="compression on .txt/.jsonl files")
    st.add_argument("--tokenizer", required=True)
    st.add_argument("files", nargs="+")
    st.set_defaults(fn=_tok_stats)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
