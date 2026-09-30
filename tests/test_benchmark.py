"""Arouse AgentBench: case files and the scorer itself."""

import pytest

torch = pytest.importorskip("torch")

from arouse.agent.episode import turn_from_event  # noqa: E402
from arouse.agent.runtime import AgentRuntime  # noqa: E402
from arouse.agent.synth import TEST_TASKS, TRAIN_TASKS, generate  # noqa: E402
from arouse.agent.tools import REGISTRY  # noqa: E402
from arouse.evaluation.agentbench import (  # noqa: E402
    eval_decisions,
    eval_end_to_end,
    gold_first_request,
    load_episodes,
    summary_lines,
)
from arouse.inference import InferenceEngine  # noqa: E402
from arouse.model.config import get_preset  # noqa: E402
from arouse.model.transformer import ArouseTransformer  # noqa: E402
from tests.conftest import ROOT  # noqa: E402
from tests.helpers import scripted_engine  # noqa: E402

TEST_FILE = ROOT / "benchmarks/agentbench_v1/test.jsonl"


def gold_turns(episodes, first_request_only=False):
    out = []
    for ep in episodes:
        evs = ep["events"]
        if first_request_only:
            prefix, _final, _ = gold_first_request(ep)
            start = len(prefix)
            for ev in evs[start:]:
                if ev["type"] == "arouse":
                    out.append(turn_from_event(ev))
                    if turn_from_event(ev).action.is_terminal:
                        break
        else:
            out += [turn_from_event(ev) for ev in evs if ev["type"] == "arouse"]
    return out


def test_benchmark_file_has_enough_held_out_cases():
    eps = load_episodes(TEST_FILE)
    decisions = sum(ev["type"] == "arouse" for ep in eps for ev in ep["events"])
    assert decisions >= 500
    assert all(ep["split"] == "test" for ep in eps)
    cats = {c for ep in eps for c in ep["category"].split("+")}
    assert {"scheduling_one_time", "scheduling_recurring", "ambiguity", "delete", "file_error", "out_of_scope"} <= cats


def test_held_out_tasks_are_unseen_in_training():
    assert not {t.lower() for t in TEST_TASKS} & {t.lower() for t in TRAIN_TASKS}
    train_tasks = {ev["turn"]["action"]["arguments"]["task"].lower()
                   for ep in generate(3000, seed=101) for ev in ep["events"]
                   if ev["type"] == "arouse" and "task" in ev["turn"]["action"].get("arguments", {})}
    assert not train_tasks & {t.lower() for t in TEST_TASKS}


def test_oracle_scores_100(tiny_tokenizer):
    """A model that outputs exactly the gold turns must get a perfect score (scorer sanity)."""
    eps = generate(25, seed=31, split="test")
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, gold_turns(eps)), REGISTRY)
    d = eval_decisions(rt, eps)
    assert d["decision_accuracy"] == 100.0 and d["structured_output_validity"] == 100.0
    assert d["response_text_exact"] == 100.0
    assert d["false_completion_rate"] in (0.0, None)
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, gold_turns(eps, first_request_only=True)), REGISTRY)
    e = eval_end_to_end(rt, eps)
    assert e["task_success_rate"] == 100.0 and e["guard_interventions"] == 0


def test_wrong_model_is_penalised(tiny_tokenizer):
    from arouse.protocol import Action, Turn

    eps = generate(10, seed=32, split="test")
    always_finish = [Turn(Action.finish("Done."))] * 200
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, always_finish), REGISTRY)
    d = eval_decisions(rt, eps)
    assert d["decision_accuracy"] < 50 and d["tool_selection_accuracy"] == 0.0


def test_untrained_model_runs_and_reports(tiny_tokenizer):
    torch.manual_seed(0)
    eng = InferenceEngine(ArouseTransformer(get_preset("arouse-tiny").replace(context_length=1024)), tiny_tokenizer)
    rt = AgentRuntime(eng, REGISTRY, max_new_tokens=16, retries=0)
    eps = generate(3, seed=33, split="test")
    report = {"decisions": eval_decisions(rt, eps, limit=4), "end_to_end": eval_end_to_end(rt, eps, limit=2)}
    assert report["decisions"]["decisions"] == 4 and report["end_to_end"]["episodes"] == 2
    assert any("Decision accuracy" in line for line in summary_lines(report))
