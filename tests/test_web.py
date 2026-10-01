"""The website runs Arouse in the browser with web/arouse.js. These tests check that the JS port
behaves like the Python implementation: same token ids, same logits (up to float rounding), same
prompts, and the same agent decisions on AgentBench cases. They run the JS in Node and skip if
Node is not installed."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arouse.agent.answers import answer_from_result  # noqa: E402
from arouse.agent.constraints import CopyConstraint  # noqa: E402
from arouse.agent.runtime import AgentRuntime  # noqa: E402
from arouse.agent.synth import episode_header, episode_sandbox  # noqa: E402
from arouse.agent.tools import REGISTRY  # noqa: E402
from arouse.agent.episode import encode_prompt  # noqa: E402
from arouse.evaluation.agentbench import load_episodes  # noqa: E402
from arouse.inference import InferenceEngine  # noqa: E402
from arouse.model.io import load_pretrained  # noqa: E402
from tests.conftest import ROOT  # noqa: E402

WEB = ROOT / "web"
MODEL = WEB / "model"
RELEASED = ROOT / "models" / "arouse-agent-s"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None or not (MODEL / "weights.bin").exists() or not RELEASED.exists(),
                                reason="needs node and the exported web model")


def run_js(tmp_path: Path, **request) -> dict:
    f = tmp_path / "req.json"
    f.write_text(json.dumps({"model_dir": str(MODEL), **request}), encoding="utf-8")
    out = subprocess.run([NODE, str(ROOT / "tests" / "web_harness.js"), str(f)], capture_output=True, text=True,
                         timeout=1800, check=True)
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def py_engine():
    """The released Python model with the website's float16 weights loaded into it."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # fast for one sequence, and no oversubscription next to Node
    model, tok, meta = load_pretrained(RELEASED)
    cfg = json.loads((MODEL / "config.json").read_text())
    assert cfg["tokenizer_fingerprint"] == tok.fingerprint()
    half = np.fromfile(MODEL / "weights.bin", dtype="<f2")
    state = model.state_dict()
    for t in cfg["tensors"]:
        n = int(np.prod(t["shape"]))
        w = torch.from_numpy(half[t["offset"]: t["offset"] + n].astype(np.float32)).view(*t["shape"])
        # the web model must be the released model (rounded to float16)
        assert torch.allclose(w, state[t["name"]].float(), atol=2e-3, rtol=1e-3), t["name"]
        state[t["name"]].copy_(w)
    yield InferenceEngine(model, tok, meta)
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def episodes():
    test = load_episodes(ROOT / "benchmarks" / "agentbench_v1" / "test.jsonl")
    ind = load_episodes(ROOT / "benchmarks" / "agentbench_v1" / "in_distribution.jsonl")
    return test[::37][:8] + ind[::31][:4]


TEXTS = [
    "Remind me tomorrow at 8 AM to check sales.",
    "Every Monday at 9:30 remind me to pay the staff 💰!",
    "  leading spaces\tand tabs\nnew lines\r\n",
    "हिन्दी में याद दिलाना, café naïve — “quotes” ’s",
    'JSON {"tool":"scheduler.create","arguments":{"task":"x"}} <|finish|> snake_case_id __init__',
    "1234567890 12:45 2026-10-05 3rd 21st",
    "日本語のテキスト 한국어 العربية",
]


def test_tokenizer_matches_python(tmp_path, py_engine, episodes):
    tok = py_engine.tokenizer
    texts = TEXTS + [e["content"] for ep in episodes for e in ep["events"] if e["type"] == "user"]
    js = run_js(tmp_path, tokenize=texts)["tokenize"]
    for t, ids in zip(texts, js, strict=True):
        assert ids == tok.encode(t), t


def test_prompts_and_logits_match_python(tmp_path, py_engine, episodes):
    tok = py_engine.tokenizer
    cases = [[ep, i] for ep in episodes[:3] for i, e in enumerate(ep["events"]) if e["type"] == "arouse"][:4]
    js = run_js(tmp_path, prompts=cases, logits=[encode_prompt(tok, episode_header(ep), ep["events"][:n])
                                                    for ep, n in cases[:2]])
    for (ep, n), ids in zip(cases, js["prompts"], strict=True):
        assert ids == encode_prompt(tok, episode_header(ep), ep["events"][:n])
    for (ep, n), lg in zip(cases[:2], js["logits"], strict=True):
        ids = torch.tensor([encode_prompt(tok, episode_header(ep), ep["events"][:n])])
        with torch.inference_mode():
            ref = py_engine.model(ids).logits[0, -1].float().numpy()
        lg = np.array(lg, dtype=np.float32)
        assert np.abs(lg - ref).max() < 2e-2 * max(1.0, np.abs(ref).max() / 10)
        assert lg.argmax() == ref.argmax()


def test_agent_decisions_match_python(tmp_path, py_engine, episodes):
    """Greedy decisions must be identical. When the Python runtime needed a resample, the two RNGs
    differ (torch vs mulberry32), so only the agreement rate is checked for those."""
    rt = AgentRuntime(py_engine, REGISTRY)
    cases = [[ep, i, mode] for ep in episodes for i, e in enumerate(ep["events"]) if e["type"] == "arouse"
             for mode in ("raw", "system")][:40]
    js = run_js(tmp_path, decisions=cases)["decisions"]
    same = greedy = 0
    for (ep, n, mode), j in zip(cases, js, strict=True):
        saved = rt.retries, rt.guard_completion, rt.constrain_copy
        if mode == "raw":
            rt.retries, rt.guard_completion, rt.constrain_copy = 0, False, False
        r = rt.next_turn(episode_header(ep), ep["events"][:n])
        rt.retries, rt.guard_completion, rt.constrain_copy = saved
        if r.attempts == 1 and j["attempts"] == 1:
            greedy += 1
            assert j["action"] == r.turn.action.to_dict(), (ep["id"], n, mode)
        same += j["action"] == r.turn.action.to_dict()
    assert greedy >= len(cases) * 0.8
    assert same >= len(cases) * 0.95


def test_end_to_end_runs_match_python(tmp_path, py_engine, episodes):
    rt = AgentRuntime(py_engine, REGISTRY)
    cases = []
    for ep in episodes:
        first_user = next(i for i, e in enumerate(ep["events"]) if e["type"] == "user")
        cases.append([ep, first_user + 1])
    js = run_js(tmp_path, episodes=cases)["episodes"]
    same = greedy = 0
    for (ep, n), j in zip(cases, js, strict=True):
        sb = episode_sandbox(ep)
        res = rt.run(episode_header(ep), ep["events"][:n], sb.execute)
        match = j["final"] == res.final.to_dict() and j["state"] == sb.snapshot()
        if all(t.attempts == 1 for t in res.turns) and all(a == 1 for a in j["attempts"]):
            greedy += 1
            assert match, ep["id"]  # no resampling: identical run
        same += match
    assert greedy >= len(cases) // 2 and same >= len(cases) * 0.75  # resampled runs use different RNGs


def test_answers_and_copy_constraint_match_python(tmp_path, py_engine):
    from arouse.agent.synth import generate

    eps = generate(60, seed=4) + generate(40, seed=5, split="test")
    prefixes = [ep["events"][:i] for ep in eps for i, e in enumerate(ep["events"]) if e["type"] == "arouse"]
    conts = []
    for ep in eps:
        ctx = episode_header(ep).context
        for i, e in enumerate(ep["events"]):
            if e["type"] == "arouse" and e["turn"]["action"]["type"] == "tool_call":
                args = e["turn"]["action"]["arguments"]
                for field in ("task", "text", "path", "date"):
                    if field in args:
                        v = args[field]
                        conts += [[ep["events"][:i], ctx, field, v[:k]] for k in (0, len(v) // 2, len(v))]
    js = run_js(tmp_path, answers=prefixes, continuations=conts)
    assert js["answers"] == [answer_from_result(p) for p in prefixes]
    tok = py_engine.tokenizer
    for (evs, ctx, field, partial), got in zip(conts, js["continuations"], strict=True):
        assert got == sorted(CopyConstraint(tok, evs, ctx).continuations(field, partial)), (field, partial)
    assert len(conts) > 100
