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
CREATE = Turn(Action.tool_call("scheduler.create", {"task": "water the plants", "date": "2026-10-01", "time": "08:00"}))
ASK = [{"type": "user", "content": "Remind me tomorrow at 8 to water the plants."}]


def test_guard_blocks_finish_after_tool_error(tiny_tokenizer):
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, [CREATE, Turn(Action.finish("Done!"))]), REGISTRY)
    sb = Sandbox(NOW, failures={"scheduler.create": 5})
    res = rt.run(HEADER, ASK, sb.execute)
    assert res.final.type == "fail" and res.turns[-1].guarded
    assert sb.snapshot()["reminders"] == []


def test_guard_allows_finish_after_success(tiny_tokenizer):
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, [CREATE, Turn(Action.finish("Done!"))]), REGISTRY)
    res = rt.run(HEADER, ASK, Sandbox(NOW).execute)
    assert res.final.type == "finish" and not res.turns[-1].guarded


def test_retry_after_error_then_success_is_allowed(tiny_tokenizer):
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, [CREATE, CREATE, Turn(Action.finish("Done"))]), REGISTRY)
    sb = Sandbox(NOW, failures={"scheduler.create": 1})
    res = rt.run(HEADER, ASK, sb.execute)
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


# --- grounding -------------------------------------------------------------------

from arouse.agent.grounding import current_request, grounding_issue  # noqa: E402


def _create(task):
    return Action.tool_call("scheduler.create", {"task": task, "date": "2026-10-01", "time": "08:00"})


def test_grounding_issue_rules():
    evs = [{"type": "user", "content": "Remind me tomorrow to Renew the  gym membership."}]
    assert grounding_issue(_create("renew the gym membership"), evs) is None  # case/space-insensitive
    assert grounding_issue(_create("renew the gym"), evs) is not None  # truncated mid-phrase: not grounded
    two = [{"type": "user", "content": "Remind me tomorrow at 8 to call the vet."}]
    assert grounding_issue(_create("call the vet"), two) is None
    assert grounding_issue(_create("call"), two) is not None
    assert grounding_issue(_create("pay rent"), [{"type": "user", "content": "remind me to pay rent every Monday at 9"}]) is None
    assert grounding_issue(_create("pay rent"), evs) is not None
    assert grounding_issue(Action.tool_call("notes.create", {"text": "renew the gym membership"}), evs) is None
    assert grounding_issue(Action.tool_call("notes.create", {"text": "the gym"}), evs) is not None
    listing = [{"type": "user", "content": "read the sales file"},
               {"type": "tool_result", "content": {"success": True, "files": ["sales_2026.csv"]}}]
    assert grounding_issue(Action.tool_call("file.read", {"path": "sales_2026.csv"}), listing) is None
    assert grounding_issue(Action.tool_call("file.read", {"path": "secret.csv"}), listing) is not None
    assert grounding_issue(Action.finish("anything"), evs) is None


def test_current_request_spans_questions_but_not_finished_requests():
    evs = [{"type": "user", "content": "old"},
           {"type": "arouse", "turn": {"action": {"type": "finish", "result": "ok"}}},
           {"type": "user", "content": "remind me to call the vet"},
           {"type": "arouse", "turn": {"action": {"type": "ask_user", "question": "When?"}}},
           {"type": "user", "content": "8 am"}]
    assert [e.get("content") for e in current_request(evs) if e["type"] == "user"] == ["remind me to call the vet", "8 am"]


def test_runtime_prefers_grounded_sample(tiny_tokenizer):
    bad = Turn(_create("renew the gym"))
    good = Turn(_create("renew the gym membership"))
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, [bad, good, Turn(Action.finish("ok"))]), REGISTRY)
    sb = Sandbox(NOW)
    res = rt.run(HEADER, [{"type": "user", "content": "Remind me tomorrow at 8 to renew the gym membership."}], sb.execute)
    assert sb.snapshot()["reminders"][0]["task"] == "renew the gym membership"
    assert res.turns[0].grounded and res.turns[0].attempts == 2


def test_runtime_keeps_greedy_when_nothing_is_grounded(tiny_tokenizer):
    bad = Turn(_create("renew the gym"))
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, [bad] * 7 + [Turn(Action.finish("ok"))]), REGISTRY)
    res = rt.run(HEADER, [{"type": "user", "content": "Remind me to renew the gym membership."}], Sandbox(NOW).execute)
    assert not res.turns[0].grounded and res.turns[0].attempts == 7


def test_raw_mode_skips_grounding(tiny_tokenizer):
    bad = Turn(_create("renew the gym"))
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, [bad, Turn(_create("x"))]), REGISTRY, retries=0)
    r = rt.next_turn(HEADER, [{"type": "user", "content": "Remind me to renew the gym membership."}])
    assert r.turn == bad and r.attempts == 1
