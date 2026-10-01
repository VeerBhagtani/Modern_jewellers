"""Arouse AgentBench: measurable agent reliability.

Two evaluations over episode files (benchmarks/agentbench_v1/*.jsonl):

1. Decisions: at every gold Arouse turn the model sees the gold history and must produce
   the next action. Two modes:
     raw    - the model alone: greedy, no constrained decoding, retries, grounding or guard
     system - the full runtime (copy-constrained decoding, validation retries, grounding, guard)
   Scores: validity, action type, tool, exact arguments, per-skill accuracy,
   false-completion rate (finish right after a tool error), ask rate on ambiguous requests.

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

from arouse.agent.answers import answer_from_result
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


def eval_decisions(runtime: AgentRuntime, episodes: list[dict[str, Any]], limit: int | None = None,
                   mode: str = "raw") -> dict[str, Any]:
    c: dict[str, int] = defaultdict(int)
    by_skill: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    failures: list[dict[str, Any]] = []
    if mode not in ("raw", "system"):
        raise ValueError("mode must be raw or system")
    saved = (runtime.retries, runtime.guard_completion, runtime.constrain_copy)
    if mode == "raw":
        runtime.retries, runtime.guard_completion, runtime.constrain_copy = 0, False, False
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
        runtime.retries, runtime.guard_completion, runtime.constrain_copy = saved
    return {
        "mode": mode,
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


def gold_answer_from_tool(ep: dict[str, Any]) -> bool:
    """True if the first request's gold final answer states a tool result (GST figures, a knowledge
    base answer, a lead list, a created reminder...), so its exact text can be checked."""
    first_user = next(i for i, e in enumerate(ep["events"]) if e["type"] == "user")
    for i in range(first_user + 1, len(ep["events"])):
        ev = ep["events"][i]
        if ev["type"] == "arouse" and turn_from_event(ev).action.is_terminal:
            a = turn_from_event(ev).action
            return a.type == "finish" and answer_from_result(ep["events"][:i]) == a.result
    return False


def eval_end_to_end(runtime: AgentRuntime, episodes: list[dict[str, Any]], limit: int | None = None) -> dict[str, Any]:
    n = ok = type_ok = state_ok = guarded = invalid = checked = answer_ok = 0
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_cat_answer: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    failures = []
    for ep in episodes[:limit]:
        prefix, gold_final, gold_state = gold_first_request(ep)
        sb = episode_sandbox(ep)
        try:
            res = runtime.run(episode_header(ep), prefix, sb.execute)
        except Exception as e:  # a crash is a failed episode, never a crashed benchmark
            n += 1
            failures.append({"episode": ep["id"], "user": prefix[-1]["content"], "error": repr(e)[:200]})
            by_cat[ep["category"].split("+")[0]][1] += 1
            continue
        t = res.final.type == gold_final.type
        s = sb.snapshot() == gold_state
        if gold_answer_from_tool(ep):
            checked += 1
            right = t and s and res.final.result == gold_final.result
            answer_ok += right
            by_cat_answer[ep["category"].split("+")[0]][0] += right
            by_cat_answer[ep["category"].split("+")[0]][1] += 1
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
        # episodes whose gold answer states a tool result: the exact answer text must match too
        "answer_correct_rate": _pct(answer_ok, checked),
        "answer_checked": checked,
        "answer_by_category": {k: {"correct": _pct(*v), "n": v[1]} for k, v in sorted(by_cat_answer.items())},
        "failures": failures,
    }


def eval_conversations(runtime: AgentRuntime, episodes: list[dict[str, Any]], limit: int | None = None) -> dict[str, Any]:
    """Whole conversations: feed the episode's user messages one by one (after each of Arouse's
    final actions), keeping Arouse's own history. Success = every reply has the gold action type,
    every tool-based answer is exactly right, and the final reminders and notes match."""
    n = ok = 0
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    failures = []
    for ep in episodes[:limit]:
        evs, sb, header = ep["events"], episode_sandbox(ep), episode_header(ep)
        history: list[dict[str, Any]] = []
        good, why = True, ""
        for i, ev in enumerate(evs):
            if ev["type"] != "user":
                continue
            j = next(k for k in range(i + 1, len(evs)) if evs[k]["type"] == "arouse" and turn_from_event(evs[k]).action.is_terminal)
            gold = turn_from_event(evs[j]).action
            history.append(ev)
            try:
                res = runtime.run(header, history, sb.execute)
            except Exception as e:  # a crash is a failed conversation
                good, why = False, repr(e)[:120]
                break
            history += res.events
            tool_answer = gold.type == "finish" and answer_from_result(evs[:j]) == gold.result
            if res.final.type != gold.type or (tool_answer and res.final.result != gold.result):
                good, why = False, f"after '{ev['content'][:60]}': got {res.final.to_dict()}"[:240]
                break
        good = good and sb.snapshot() == ep["final_state"]
        n += 1
        ok += good
        for cat in set(ep["category"].split("+")):
            by_cat[cat][0] += good
            by_cat[cat][1] += 1
        if not good and len(failures) < 25:
            failures.append({"episode": ep["id"], "category": ep["category"], "why": why or "final state differs"})
    return {"episodes": n, "conversation_success_rate": _pct(ok, n),
            "by_category": {k: {"success": _pct(*v), "n": v[1]} for k, v in sorted(by_cat.items())}, "failures": failures}


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
        "decisions": eval_decisions(runtime, eps, decision_limit, mode="raw"),
        "decisions_system": eval_decisions(runtime, eps, decision_limit, mode="system"),
        "end_to_end": eval_end_to_end(runtime, eps, e2e_limit),
        "conversations": eval_conversations(runtime, eps, e2e_limit),
    }
    report["seconds"] = round(time.perf_counter() - t0, 1)
    return report


def summary_lines(report: dict[str, Any]) -> list[str]:
    d, e = report["decisions"], report["end_to_end"]
    ds = report.get("decisions_system", d)
    rows = [
        ("Decisions evaluated", d["decisions"], ds["decisions"]),
        ("Decision accuracy (type + exact tool/args)", d["decision_accuracy"], ds["decision_accuracy"]),
        ("Structured output validity", d["structured_output_validity"], ds["structured_output_validity"]),
        ("Action type accuracy", d["action_type_accuracy"], ds["action_type_accuracy"]),
        ("Tool selection accuracy", d["tool_selection_accuracy"], ds["tool_selection_accuracy"]),
        ("Argument exact-match accuracy", d["argument_exact_accuracy"], ds["argument_exact_accuracy"]),
        ("Scheduling exact accuracy", d["scheduling_exact_accuracy"], ds["scheduling_exact_accuracy"]),
        ("Ask rate on ambiguous requests", d["ask_rate_on_ambiguous"], ds["ask_rate_on_ambiguous"]),
        ("False completion rate", d["false_completion_rate"], ds["false_completion_rate"]),
    ]
    fmt = lambda v: f"{v}%" if isinstance(v, float) else str(v)  # noqa: E731
    out = [f"{'metric':<46} {'raw model':>10} {'system':>10}"]
    out += [f"{k:<46} {fmt(a):>10} {fmt(b):>10}" for k, a, b in rows]
    out.append(f"{'End-to-end task success (system)':<46} {'':>10} {fmt(e['task_success_rate']):>10}  (n={e['episodes']})")
    if "conversations" in report:
        c = report["conversations"]
        out.append(f"{'Whole conversations right (system)':<46} {'':>10} {fmt(c['conversation_success_rate']):>10}  (n={c['episodes']})")
    if e.get("answer_checked"):
        out.append(f"{'Answer exactly right (tool-based answers)':<46} {'':>10} {fmt(e['answer_correct_rate']):>10}  (n={e['answer_checked']})")
    out.append("By skill (raw): " + ", ".join(f"{k} {v['accuracy']}% (n={v['n']})" for k, v in d["by_skill"].items()))
    return out
