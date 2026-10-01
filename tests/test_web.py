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
from arouse.agent.grounding import answer_issue, grounding_issue  # noqa: E402
from arouse.protocol import Action  # noqa: E402
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
                for field in ("task", "text", "query", "path", "date", "time", "amount", "rate"):
                    if field in args:
                        v = args[field]
                        conts += [[ep["events"][:i], ctx, field, v[:k]] for k in (0, len(v) // 2, len(v))]
    actions, completes = [], []
    for ep in eps:
        for i, e in enumerate(ep["events"]):
            a = e["type"] == "arouse" and e["turn"]["action"]
            if a and a["type"] == "tool_call":
                actions.append([a, ep["events"][:i]])
                if a["tool"] == "scheduler.delete":  # also every other id, which must be refused
                    actions += [[{**a, "arguments": {"task_id": f"r-{k}"}}, ep["events"][:i]] for k in range(1, 5)]
                for field in ("task", "text", "query", "path", "date", "time", "amount", "rate"):
                    if field in a["arguments"]:
                        v = a["arguments"][field]
                        completes += [[ep["events"][:i], episode_header(ep).context, field, v[:k]] for k in (len(v) // 2, len(v))]
    texts = []
    for ep in eps:
        for i, e in enumerate(ep["events"]):
            a = e["type"] == "arouse" and e["turn"]["action"]
            if a and a["type"] != "tool_call":
                texts.append([a, ep["events"][:i]])
                if a["type"] == "fail" and "couldn't find a reminder to" in a["error"]:  # wrong claims must be caught too
                    texts += [[{**a, "error": a["error"].replace("to ", "to the ", 1)}, ep["events"][:i]]]
                if a["type"] == "finish":
                    texts.append([{**a, "result": a["result"] + " Also the cow."}, ep["events"][:i]])
    js = run_js(tmp_path, answers=prefixes, continuations=conts, grounding=actions, complete=completes, answer_issue=texts)
    assert js["answer_issue"] == [answer_issue(Action.from_dict(a), evs) is not None for a, evs in texts]
    assert sum(js["answer_issue"]) > 20
    assert js["answers"] == [answer_from_result(p) for p in prefixes]
    assert js["grounding"] == [grounding_issue(Action.from_dict(a), evs) is not None for a, evs in actions]
    assert js["complete"] == [CopyConstraint(py_engine.tokenizer, evs, ctx).complete(f, v) for evs, ctx, f, v in completes]
    assert sum(a["tool"] == "scheduler.delete" for a, _ in actions) >= 20
    tok = py_engine.tokenizer
    for (evs, ctx, field, partial), got in zip(conts, js["continuations"], strict=True):
        assert got == sorted(CopyConstraint(tok, evs, ctx).continuations(field, partial)), (field, partial)
    assert len(conts) > 100


def test_website_runs_the_model_in_the_browser():
    """Serve web/ statically, load it in Chromium, and complete one real request."""
    import functools
    import http.server
    import threading

    sync_api = pytest.importorskip("playwright.sync_api")
    exe = next((c for c in ("/opt/pw-browsers/chromium", shutil.which("chromium")) if c and Path(c).exists()), None)
    if exe is None:
        pytest.skip("no Chromium binary")

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):  # noqa: ANN002
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with sync_api.sync_playwright() as p:
            browser = p.chromium.launch(executable_path=exe)
            pg = browser.new_page(viewport={"width": 390, "height": 760})
            errors: list[str] = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(f"http://127.0.0.1:{srv.server_port}/")
            pg.wait_for_selector("#status.ok", timeout=120000)
            pg.fill("#input", "Every Monday at 9 AM remind me to pay the staff.")
            pg.keyboard.press("Enter")
            pg.wait_for_selector(".msg.agent .answer", timeout=120000)
            assert "pay the staff" in pg.inner_text(".msg.agent .answer")
            pg.click("#tab-space")
            assert "pay the staff" in pg.inner_text("#reminders") and "Weekly" in pg.inner_text("#reminders")
            assert pg.evaluate("document.documentElement.scrollWidth") <= 390  # no sideways scroll on a phone
            pg.reload()
            pg.wait_for_selector("#status.ok", timeout=120000)
            assert pg.locator(".msg.user").count() == 1  # the conversation survives a reload
            assert not errors
            browser.close()
    finally:
        srv.shutdown()
        srv.server_close()


def test_new_tools_match_python_exactly(tmp_path):
    """Knowledge search, GST and leads: every gold tool call replayed in the JS sandbox gives the same result."""
    import random

    from arouse.agent.knowledge import load as load_kb
    from arouse.agent.knowledge import search
    from arouse.agent.synth import TEST_KB_QUESTIONS, TEST_UNKNOWN, generate, unknown_question

    eps = [ep for ep in generate(500, seed=61) + generate(200, seed=62, split="test")
           if any(c.startswith(("leads", "gst", "question")) for c in ep["category"].split("+"))]
    rng = random.Random(5)
    queries = [q for e in load_kb()["entries"] for q in [e["q"], *e["alts"]]] + [q for _, q in TEST_KB_QUESTIONS]
    queries += TEST_UNKNOWN + [unknown_question(rng) for _ in range(200)] + ["", "e-way bill?", "GSTR-3B", "22K vs 24K gold"]
    js = run_js(tmp_path, replay=eps, kb=queries)
    for ep, results in zip(eps, js["replay"], strict=True):
        assert results == [ev for ev in ep["events"] if ev["type"] in ("tool_result", "tool_error")], ep["id"]
    assert js["kb"] == [search(q) for q in queries]
    assert len(eps) > 150


def test_amount_rate_time_mentions_match_python(tmp_path):
    from arouse.agent.mentions import gst_mentions, mentioned_times
    from arouse.agent.synth import generate

    eps = generate(400, seed=63) + generate(200, seed=64, split="test")
    texts = [e["content"] for ep in eps for e in ep["events"] if e["type"] == "user"]
    texts += ["at 9:30 at night", "12 AM", "at 0:30", "13 pm", "at 7.", "8:30 or 6 PM", "noon or midnight", "a.m. 9"]
    requests = [ep["events"][:i] for ep in eps for i, e in enumerate(ep["events"]) if e["type"] == "arouse"]
    js = run_js(tmp_path, times=texts, gst=requests)
    assert js["times"] == [sorted(mentioned_times([t])) for t in texts]
    assert js["gst"] == [[sorted(x) for x in gst_mentions(r)] for r in requests]
