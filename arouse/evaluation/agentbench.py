"""Arouse AgentBench: measurable agent reliability.

Two evaluations over episode files (benchmarks/agentbench_v1/*.jsonl):

1. Decisions (model only, greedy, no retries, no guard): at every gold Arouse turn the
   model sees the gold history and must produce the next action. Scores:
   validity, action type, tool, exact arguments, per-skill accuracy, false-completion
   rate (finish right after a tool error), ask rate on ambiguous requests.

2. End-to-end (full runtime incl. retries + completion guard) on each episode's first
   request: run against a fresh sandbox; success = same terminal action type as gold AND
   identical resulting reminders/notes.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from arouse.agent.episode import turn_from_event
from arouse.agent.runtime import AgentRuntime, last_observation
from arouse.agent.synth import episode_header, episode_sandbox
from arouse.agent.tools import REGISTRY
from arouse.inference import InferenceEngine
from arouse.protocol import Action

SKILLS = ("intent", "tool_selection", "argument_generation", "scheduling", "multi_step", "task_state",
          "error_recovery", "ambiguity", "completion_verification", "structured_output")


def load_episodes(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def action_correct(pred: Action, gold: Action) -> bool:
    if pred.type != gold.type:
        return False
    if gold.type == "tool_call":
        return pred.tool == gold.tool and pred.arguments == gold.arguments
    return True


def _pct(n: int, d: int) -> float | None:
    return round(100.0 * n / d, 2) if d else None


def eval_decisions(runtime: AgentRuntime, episodes: list[dict[str, Any]], limit: int | None = None) -> dict[str, Any]:
    c: dict[str, int] = defaultdict(int)
    by_skill: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    failures: list[dict[str, Any]] = []
    saved = (runtime.retries, runtime.guard_completion)
    runtime.retries, runtime.guard_completion = 0, False  # raw model behaviour
    try:
        for ep in episodes:
            header = episode_header(ep)
            k = -1
            for i, ev in enumerate(ep["events"]):
                if ev["type"] != "arouse":
                    continue
                k += 1
                if limit is not None and c["decisions"] >= limit:
                    break
                gold = turn_from_event(ev).action
                prefix = ep["events"][:i]
                r = runtime.next_turn(header, prefix)
                pred = r.turn.action
                c["decisions"] += 1
                c["valid"] += r.valid
                c["type_ok"] += r.valid and pred.type == gold.type
                ok = r.valid and action_correct(pred, gold)
                c["correct"] += ok
                if gold.type == "tool_call":
                    c["gold_tool_calls"] += 1
                    c["tool_ok"] += r.valid and pred.type == "tool_call" and pred.tool == gold.tool
                    c["args_ok"] += ok
                    if gold.tool == "scheduler.create":
                        c["sched"] += 1
                        c["sched_ok"] += ok
                if last_observation(prefix) == "tool_error":
                    c["after_error"] += 1
                    c["false_completion"] += r.valid and pred.type == "finish"
                if gold.type == "ask_user" and "ambiguity" in ep["skills"][k]:
                    c["ambiguous"] += 1
                    c["asked"] += r.valid and pred.type == "ask_user"
                if gold.type in ("finish", "ask_user", "fail") and r.valid and pred.type == gold.type:
                    c["text_cmp"] += 1
                    c["text_exact"] += pred.to_dict() == gold.to_dict()
                for s in ep["skills"][k]:
                    by_skill[s][0] += ok
                    by_skill[s][1] += 1
                for cat in ep["category"].split("+"):
                    by_cat[cat][0] += ok
                    by_cat[cat][1] += 1
                if not ok and len(failures) < 40:
                    failures.append({"episode": ep["id"], "turn": k, "gold": gold.to_dict(),
                                     "pred": pred.to_dict() if r.valid else None, "raw": r.raw_text[:300]})
    finally:
        runtime.retries, runtime.guard_completion = saved
    return {
        "decisions": c["decisions"],
        "decision_accuracy": _pct(c["correct"], c["decisions"]),
        "structured_output_validity": _pct(c["valid"], c["decisions"]),
        "action_type_accuracy": _pct(c["type_ok"], c["decisions"]),
        "tool_selection_accuracy": _pct(c["tool_ok"], c["gold_tool_calls"]),
        "argument_exact_accuracy": _pct(c["args_ok"], c["gold_tool_calls"]),
        "scheduling_exact_accuracy": _pct(c["sched_ok"], c["sched"]),
        "ask_rate_on_ambiguous": _pct(c["asked"], c["ambiguous"]),
        "false_completion_rate": _pct(c["false_completion"], c["after_error"]),
        "response_text_exact": _pct(c["text_exact"], c["text_cmp"]),
        "counts": dict(c),
        "by_skill": {s: {"accuracy": _pct(*by_skill[s]), "n": by_skill[s][1]} for s in SKILLS if by_skill[s][1]},
        "by_category": {k: {"accuracy": _pct(*v), "n": v[1]} for k, v in sorted(by_cat.items())},
        "failures": failures,
    }


def gold_first_request(ep: dict[str, Any]) -> tuple[list[dict[str, Any]], Action, dict[str, Any]]:
    """(events up to the first user message, gold terminal action, gold sandbox state after it)."""
    sb = episode_sandbox(ep)
    first_user = next(i for i, e in enumerate(ep["events"]) if e["type"] == "user")
    for ev in ep["events"][first_user + 1:]:
        if ev["type"] == "arouse":
            a = turn_from_event(ev).action
            if a.is_terminal:
                return ep["events"][: first_user + 1], a, sb.snapshot()
            sb.execute(a.tool, a.arguments)
    raise ValueError(f"episode {ep['id']} has no terminal action")


def eval_end_to_end(runtime: AgentRuntime, episodes: list[dict[str, Any]], limit: int | None = None) -> dict[str, Any]:
    n = ok = type_ok = state_ok = guarded = invalid = 0
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    failures = []
    for ep in episodes[:limit]:
        prefix, gold_final, gold_state = gold_first_request(ep)
        sb = episode_sandbox(ep)
        res = runtime.run(episode_header(ep), prefix, sb.execute)
        t = res.final.type == gold_final.type
        s = sb.snapshot() == gold_state
        n += 1
        type_ok += t
        state_ok += s
        ok += t and s
        guarded += any(tr.guarded for tr in res.turns)
        invalid += any(not tr.valid for tr in res.turns)
        cat = ep["category"].split("+")[0]
        by_cat[cat][0] += t and s
        by_cat[cat][1] += 1
        if not (t and s) and len(failures) < 25:
            failures.append({"episode": ep["id"], "user": prefix[-1]["content"], "gold": gold_final.to_dict(),
                             "pred": res.final.to_dict()})
    return {
        "episodes": n,
        "task_success_rate": _pct(ok, n),
        "final_action_type_accuracy": _pct(type_ok, n),
        "state_correct_rate": _pct(state_ok, n),
        "guard_interventions": guarded,
        "episodes_with_invalid_output": invalid,
        "by_category": {k: {"success": _pct(*v), "n": v[1]} for k, v in sorted(by_cat.items())},
        "failures": failures,
    }


def run_benchmark(model_dir: str, bench_file: str, *, decision_limit: int | None = None,
                  e2e_limit: int | None = None) -> dict[str, Any]:
    engine = InferenceEngine.from_pretrained(model_dir)
    runtime = AgentRuntime(engine, REGISTRY)
    eps = load_episodes(bench_file)
    t0 = time.perf_counter()
    report = {
        "benchmark": "arouse-agentbench-v1",
        "file": str(bench_file),
        "model": engine.info(),
        "decisions": eval_decisions(runtime, eps, decision_limit),
        "end_to_end": eval_end_to_end(runtime, eps, e2e_limit),
    }
    report["seconds"] = round(time.perf_counter() - t0, 1)
    return report


def summary_lines(report: dict[str, Any]) -> list[str]:
    d, e = report["decisions"], report["end_to_end"]
    rows = [
        ("Decisions evaluated", d["decisions"]),
        ("Decision accuracy (type + exact tool/args)", d["decision_accuracy"]),
        ("Structured output validity", d["structured_output_validity"]),
        ("Action type accuracy", d["action_type_accuracy"]),
        ("Tool selection accuracy", d["tool_selection_accuracy"]),
        ("Argument exact-match accuracy", d["argument_exact_accuracy"]),
        ("Scheduling exact accuracy", d["scheduling_exact_accuracy"]),
        ("Ask rate on ambiguous requests", d["ask_rate_on_ambiguous"]),
        ("False completion rate (raw model)", d["false_completion_rate"]),
        ("End-to-end task success", e["task_success_rate"]),
        ("End-to-end episodes", e["episodes"]),
    ]
    out = [f"{k:<46} {v}{'%' if isinstance(v, float) else ''}" for k, v in rows]
    out.append("By skill: " + ", ".join(f"{k} {v['accuracy']}% (n={v['n']})" for k, v in d["by_skill"].items()))
    return out
