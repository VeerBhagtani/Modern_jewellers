"""Verification: requested != completed. False "Done." must never reach the user."""

from datetime import datetime

from arouse.agent.context import build_context
from arouse.agent.episode import Header, turn_from_event
from arouse.agent.runtime import AgentRuntime, last_observation
from arouse.agent.synth import generate
from arouse.agent.tools import REGISTRY, Sandbox
from arouse.protocol import Action, Turn
from arouse.tokenizer import Special
from tests.helpers import scripted_engine

NOW = datetime(2026, 9, 30, 19, 21)
HEADER = Header(context=build_context(NOW), tools=REGISTRY.names())
CREATE = Turn(Action.tool_call("scheduler.create", {"task": "a", "date": "2026-10-01", "time": "08:00"}))


def test_guard_blocks_finish_after_tool_error(tiny_tokenizer):
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, [CREATE, Turn(Action.finish("Done!"))]), REGISTRY)
    sb = Sandbox(NOW, failures={"scheduler.create": 5})
    res = rt.run(HEADER, [{"type": "user", "content": "remind me"}], sb.execute)
    assert res.final.type == "fail" and res.turns[-1].guarded
    assert sb.snapshot()["reminders"] == []


def test_guard_allows_finish_after_success(tiny_tokenizer):
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, [CREATE, Turn(Action.finish("Done!"))]), REGISTRY)
    res = rt.run(HEADER, [{"type": "user", "content": "remind me"}], Sandbox(NOW).execute)
    assert res.final.type == "finish" and not res.turns[-1].guarded


def test_retry_after_error_then_success_is_allowed(tiny_tokenizer):
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, [CREATE, CREATE, Turn(Action.finish("Done"))]), REGISTRY)
    sb = Sandbox(NOW, failures={"scheduler.create": 1})
    res = rt.run(HEADER, [{"type": "user", "content": "remind me"}], sb.execute)
    assert [e["type"] for e in res.events] == ["arouse", "tool_error", "arouse", "tool_result", "arouse"]
    assert res.final.type == "finish" and len(sb.snapshot()["reminders"]) == 1


def test_invalid_model_output_becomes_honest_fail(tiny_tokenizer):
    garbage = [Special.FINISH, *tiny_tokenizer.encode("{not json"), Special.END]
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, [garbage]), REGISTRY, retries=2)
    res = rt.run(HEADER, [{"type": "user", "content": "hi"}], Sandbox(NOW).execute)
    assert res.final.type == "fail" and not res.turns[0].valid and res.turns[0].attempts == 3


def test_schema_invalid_tool_call_is_resampled_not_executed(tiny_tokenizer):
    bad = Turn(Action.tool_call("scheduler.create", {"task": "a", "time": "8am"}))  # time format invalid
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, [bad, bad, bad]), REGISTRY, retries=2)
    sb = Sandbox(NOW)
    res = rt.run(HEADER, [{"type": "user", "content": "remind me"}], sb.execute)
    assert res.final.type == "fail" and sb.snapshot()["reminders"] == []


def test_step_limit(tiny_tokenizer):
    lst = Turn(Action.tool_call("scheduler.list", {}))
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, [lst] * 20), REGISTRY, max_tool_calls=3)
    res = rt.run(HEADER, [{"type": "user", "content": "loop"}], Sandbox(NOW).execute)
    assert res.final.type == "fail" and sum(e["type"] == "tool_result" for e in res.events) == 3


def test_last_observation_is_scoped_to_current_request():
    evs = [{"type": "user", "content": "a"}, {"type": "tool_error", "content": {}}, {"type": "user", "content": "b"}]
    assert last_observation(evs) is None
    assert last_observation(evs[:2]) == "tool_error"


def test_training_data_never_finishes_after_an_error():
    finishes = 0
    for ep in generate(2000, seed=21):
        evs = ep["events"]
        for i, ev in enumerate(evs):
            if ev["type"] == "arouse" and turn_from_event(ev).action.type == "finish":
                finishes += 1
                assert last_observation(evs[:i]) != "tool_error", ep["id"]
    assert finishes > 1000
