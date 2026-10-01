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
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, [bad, good, Turn(Action.finish("ok"))]), REGISTRY, constrain_copy=False)
    sb = Sandbox(NOW)
    res = rt.run(HEADER, [{"type": "user", "content": "Remind me tomorrow at 8 to renew the gym membership."}], sb.execute)
    assert sb.snapshot()["reminders"][0]["task"] == "renew the gym membership"
    assert res.turns[0].grounded and res.turns[0].attempts == 2


def test_runtime_keeps_greedy_when_nothing_is_grounded(tiny_tokenizer):
    bad = Turn(_create("renew the gym"))
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, [bad] * 7 + [Turn(Action.finish("ok"))]), REGISTRY, constrain_copy=False)
    res = rt.run(HEADER, [{"type": "user", "content": "Remind me to renew the gym membership."}], Sandbox(NOW).execute)
    assert not res.turns[0].grounded and res.turns[0].attempts == 7


def test_raw_mode_skips_grounding(tiny_tokenizer):
    bad = Turn(_create("renew the gym"))
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, [bad, Turn(_create("x"))]), REGISTRY, retries=0, constrain_copy=False)
    r = rt.next_turn(HEADER, [{"type": "user", "content": "Remind me to renew the gym membership."}])
    assert r.turn == bad and r.attempts == 1


# --- constrained decoding ----------------------------------------------------------

from datetime import date  # noqa: E402

from arouse.agent.constraints import CopyConstraint, mentioned_dates  # noqa: E402


def _allowed_texts(tok, cc, prefix_text):
    import torch

    ids = tok.encode(prefix_text, allow_special=True)
    out = cc(ids, torch.zeros(tok.vocab_size))
    return {tok.id_to_bytes(i).decode("utf-8", "ignore") for i in torch.isfinite(out).nonzero().flatten().tolist()}


def test_copy_constraint_only_allows_user_phrases(tiny_tokenizer):
    evs = [{"type": "user", "content": "Remind me tomorrow at 8 to test the backup generator."}]
    cc = CopyConstraint(tiny_tokenizer, evs, build_context(NOW))
    head = '<|tool_call|>{"tool":"scheduler.create","arguments":{"date":"2026-10-01","task":"test the'
    allowed = _allowed_texts(tiny_tokenizer, cc, head)
    assert allowed and all(" backup generator.".startswith(p) for p in allowed)  # only the real continuation
    done = _allowed_texts(tiny_tokenizer, cc, head + ' backup generator')
    assert any(p.startswith('"') for p in done)  # may close once the phrase is complete
    assert not any(p.startswith('"') for p in _allowed_texts(tiny_tokenizer, cc, head + " back"))  # not mid-word


def test_copy_constraint_inactive_outside_copy_fields(tiny_tokenizer):
    import torch

    cc = CopyConstraint(tiny_tokenizer, [{"type": "user", "content": "hi"}], build_context(NOW))
    logits = torch.randn(tiny_tokenizer.vocab_size)
    ids = tiny_tokenizer.encode('<|plan|>thinking<|finish|>{"result":"', allow_special=True)
    assert torch.equal(cc(ids, logits), logits)


def test_date_constraint(tiny_tokenizer):
    ctx = build_context(NOW)  # tomorrow = 2026-10-01, next friday = 2026-10-02
    evs = [{"type": "user", "content": "Remind me on October 20 at 9 to pay rent."}]
    cc = CopyConstraint(tiny_tokenizer, evs, ctx)
    assert cc.dates == ["2026-10-20"]  # only the date the user named
    allowed = _allowed_texts(tiny_tokenizer, cc, '<|tool_call|>{"tool":"scheduler.create","arguments":{"date":"2026-10-')
    assert allowed and all(any(d[8:].startswith(p) for d in cc.dates if d.startswith("2026-10-")) for p in allowed)


def test_mentioned_dates():
    today = date(2026, 9, 30)
    got = mentioned_dates(["on October 5", "the 3rd of Jan", "12 Sept", "Feb 30", "may I ask"], today)
    assert got == {"2026-10-05", "2027-01-03", "2027-09-12"}


def test_copy_constraint_stops_and_schema_order(tiny_tokenizer):
    evs = [{"type": "user", "content": "Give me a reminder to renew the gym membership in 3 hours."}]
    cc = CopyConstraint(tiny_tokenizer, evs, build_context(NOW))
    assert cc.continuations("task", "renew the gym") == [" membership"]  # cannot run into " in 3 hours"
    assert all(not c.startswith("3") for c in cc.continuations("task", ""))  # tasks never start with a digit
    assert CopyConstraint.next_after_close("task", '{"tool":"scheduler.create","arguments":{"date":"x","task":"') == ","
    assert CopyConstraint.next_after_close("task", '{"tool":"scheduler.create","arguments":{"in_minutes":5,"task":"') == "}"
    assert CopyConstraint.next_after_close("text", '{"tool":"notes.create","arguments":{"text":"') == "}"


def test_answer_guard_catches_invented_words_but_never_gold_answers():
    from arouse.agent.grounding import answer_issue

    res = {"success": True, "count": 1, "reminders": [{"task_id": "r-2", "task": "pay the staff", "next_run": "2026-10-05T09:00"}]}
    evs = [{"type": "user", "content": "What reminders do I have?"},
           {"type": "arouse", "turn": {"action": {"type": "tool_call", "tool": "scheduler.list", "arguments": {}}}},
           {"type": "tool_result", "content": res}]
    assert answer_issue(Action.finish("You have 1 reminder: pay the staff on 2026-10-05 at 09:00."), evs) is None
    assert "cow" in answer_issue(Action.finish("You have 1 reminder: call the cow on 2026-10-05 at 09:00."), evs)
    assert answer_issue(Action.finish("Reminder set: x on 2027-01-01 at 09:00."), evs) is not None  # date not seen
    flagged = 0
    for ep in generate(800, seed=779) + generate(200, seed=780, split="test"):
        for i, ev in enumerate(ep["events"]):
            if ev["type"] == "arouse":
                flagged += answer_issue(turn_from_event(ev).action, ep["events"][:i]) is not None
    assert flagged == 0


def test_path_constraint_only_allows_whole_file_names(tiny_tokenizer):
    evs = [{"type": "user", "content": "Count the lines in vet_visits.txt."}]
    cc = CopyConstraint(tiny_tokenizer, evs, build_context(NOW))
    assert cc.continuations("path", "") == ["vet_visits.txt"]  # not "the", "lines", "visits.txt" or "txt"
    listed = evs + [{"type": "arouse", "turn": {"action": {"type": "tool_call", "tool": "file.list", "arguments": {}}}},
                    {"type": "tool_result", "content": {"success": True, "files": ["herd 2026.csv", "sales.csv"]}}]
    cc = CopyConstraint(tiny_tokenizer, listed, build_context(NOW))
    assert set(cc.continuations("path", "")) == {"vet_visits.txt", "herd 2026.csv", "sales.csv"}


def test_answer_from_result_matches_training_replies():
    from arouse.agent.answers import answer_from_result

    checked = 0
    for ep in generate(600, seed=31):
        for i, ev in enumerate(ep["events"]):
            a = ev["type"] == "arouse" and ev["turn"]["action"]
            prev = ep["events"][i - 1] if i else {}
            if a and a["type"] == "finish" and prev.get("type") == "tool_result":
                tool = ep["events"][i - 2]["turn"]["action"]["tool"]
                got = answer_from_result(ep["events"][:i])
                if tool == "file.read" and " It starts with" not in a["result"]:
                    got = got.split(" It starts with")[0]  # "how many lines" replies leave out the preview
                assert got == a["result"], (tool, a["result"])
                checked += 1
    assert checked > 200
    assert answer_from_result([{"type": "user", "content": "hi"}]) is None


def test_ungrounded_final_answer_is_written_from_the_tool_result(tiny_tokenizer):
    invented = Turn(Action.finish("Reminder set: water the orchids on 2026-10-01 at 08:00."))
    rt = AgentRuntime(scripted_engine(tiny_tokenizer, [CREATE] + [invented] * 7), REGISTRY, constrain_copy=False)
    res = rt.run(HEADER, ASK, Sandbox(NOW).execute)
    assert res.final.result == "Reminder set: water the plants on 2026-10-01 at 08:00."
    assert res.turns[-1].rewritten and "tool result" in res.turns[-1].turn.verify
    honest = Turn(Action.finish("Reminder set: water the plants on 2026-10-01 at 08:00."))
    res = AgentRuntime(scripted_engine(tiny_tokenizer, [CREATE, honest]), REGISTRY, constrain_copy=False).run(
        HEADER, ASK, Sandbox(NOW).execute)
    assert res.final == honest.action and not res.turns[-1].rewritten


def test_dates_follow_the_words_of_the_request(tiny_tokenizer):
    from arouse.agent.constraints import referenced_dates

    now = NOW  # Wednesday 2026-09-30 19:21
    assert referenced_dates(["Remind me on Friday to order feed.", "6 pm"], now) == {"2026-10-02"}
    assert referenced_dates(["remind me this wednesday"], now) == {"2026-10-07"}  # today is Wednesday: next week
    assert referenced_dates(["tonight at 9", "and tomorrow"], now) == {"2026-09-30", "2026-10-01"}
    assert referenced_dates(["the day after tomorrow at 9"], now) == {"2026-10-02"}
    assert referenced_dates(["at 5 pm remind me to call mom"], now) == set()
    ask = {"type": "arouse", "turn": {"action": {"type": "ask_user",
                                                 "question": "19:00 today has already passed. Should I set it for tomorrow at 19:00 instead?"}}}
    evs = [{"type": "user", "content": "Remind me today at 7 PM to feed the dog."}, ask, {"type": "user", "content": "yes"}]
    assert CopyConstraint(tiny_tokenizer, evs, build_context(NOW)).dates == ["2026-09-30", "2026-10-01"]
    no_day = [{"type": "user", "content": "Remind me at 5 to call mom."}]
    assert len(CopyConstraint(tiny_tokenizer, no_day, build_context(NOW)).dates) == 8  # whole calendar (today + 7 days)


def _listed(*tasks):
    rems = [{"task_id": f"r-{i + 1}", "task": t, "next_run": "2026-10-05T09:00"} for i, t in enumerate(tasks)]
    return [{"type": "arouse", "turn": {"action": {"type": "tool_call", "tool": "scheduler.list", "arguments": {}}}},
            {"type": "tool_result", "content": {"success": True, "count": len(rems), "reminders": rems}}]


def test_delete_must_target_the_named_reminder():
    from arouse.agent.grounding import delete_issue

    evs = [{"type": "user", "content": "Please drop the call the vet reminder."}] + _listed("call the accountant", "call the vet")
    assert delete_issue("r-2", evs) is None
    assert delete_issue("r-1", evs) is not None  # "call" matches, but "call the vet" matches better
    assert delete_issue("r-9", evs) is not None  # not listed
    other = [{"type": "user", "content": "I no longer need the reminder to order spare pump parts."}] + _listed("defrost the freezer")
    assert delete_issue("r-1", other) is not None  # nothing the user said matches
    earlier = [{"type": "user", "content": "Remind me tomorrow at 8 to check sales."},
               {"type": "arouse", "turn": {"action": {"type": "finish", "result": "Reminder set."}}},
               {"type": "user", "content": "Actually cancel that."}] + _listed("check sales", "pay rent")
    assert delete_issue("r-1", earlier) is None and delete_issue("r-2", earlier) is not None  # "that" = the earlier request


def test_runtime_never_deletes_an_unnamed_reminder(tiny_tokenizer):
    from datetime import datetime as dt

    from arouse.agent.tools import Reminder

    wrong = Turn(Action.tool_call("scheduler.delete", {"task_id": "r-1"}))
    eng = scripted_engine(tiny_tokenizer, [Turn(Action.tool_call("scheduler.list", {}))] + [wrong] * 7)
    sb = Sandbox(NOW, reminders=[Reminder("r-1", "defrost the freezer", dt(2026, 10, 3, 9, 0))])
    res = AgentRuntime(eng, REGISTRY, constrain_copy=False).run(
        HEADER, [{"type": "user", "content": "Delete my reminder to order spare pump parts."}], sb.execute)
    assert res.final.type == "fail" and len(sb.snapshot()["reminders"]) == 1


def test_notes_are_copied_to_the_end_of_the_sentence(tiny_tokenizer):
    evs = [{"type": "user", "content": "Please jot down: the market is closed on Sunday, thanks"}]
    cc = CopyConstraint(tiny_tokenizer, evs, build_context(NOW))
    assert not cc.complete("text", "the market is closed")
    assert cc.complete("text", "the market is closed on Sunday")
    price = [{"type": "user", "content": "Note that milk is 4.50 per litre now."}]
    cc = CopyConstraint(tiny_tokenizer, price, build_context(NOW))
    assert "milk is 4.50 per litre now" in [("milk" + c) for c in cc.continuations("text", "milk")]


def test_labelled_note_is_the_content_after_the_label(tiny_tokenizer):
    from arouse.agent.constraints import label_end

    assert label_end("Please jot down: the pump makes a strange noise.") == len("Please jot down: ")
    assert label_end("Note that the meeting is at 10:30.") is None  # no "label: " (10:30 is a time)
    evs = [{"type": "user", "content": "Remember this note: the pump makes a strange noise."}]
    cc = CopyConstraint(tiny_tokenizer, evs, build_context(NOW))
    assert cc.continuations("text", "") == ["the pump makes a strange noise"]
    assert not cc.complete("text", "Remember this note: the pump makes a strange noise")
    assert not cc.complete("text", "a strange noise") and cc.complete("text", "the pump makes a strange noise")


def test_confirmation_must_state_the_tool_result(tiny_tokenizer):
    from arouse.agent.answers import corrected_confirmation

    garbled = Turn(Action.finish("Reminder set: water the your files on 2026-10-01 at 08:00."))
    res = AgentRuntime(scripted_engine(tiny_tokenizer, [CREATE, garbled]), REGISTRY, constrain_copy=False).run(
        HEADER, ASK, Sandbox(NOW).execute)
    assert res.final.result == "Reminder set: water the plants on 2026-10-01 at 08:00." and res.turns[-1].rewritten
    raw = AgentRuntime(scripted_engine(tiny_tokenizer, [CREATE, garbled]), REGISTRY, retries=0, constrain_copy=False).run(
        HEADER, ASK, Sandbox(NOW).execute)
    assert raw.final == garbled.action  # raw mode reports the model's own text
    lines = [{"type": "user", "content": "How many lines are in a.csv?"},
             {"type": "arouse", "turn": {"action": {"type": "tool_call", "tool": "file.read", "arguments": {"path": "a.csv"}}}},
             {"type": "tool_result", "content": {"success": True, "path": "a.csv", "lines": 3, "preview": "x,y\n1,2"}}]
    assert corrected_confirmation(Action.finish("a.csv has 3 lines."), lines) is None
    assert corrected_confirmation(Action.finish("a.csv has 4 lines."), lines) == "a.csv has 3 lines. It starts with: x,y"
    assert corrected_confirmation(Action.fail("x"), lines) is None


def test_not_found_claims_must_agree_with_the_tools():
    from arouse.agent.grounding import answer_issue

    evs = [{"type": "user", "content": "Cancel my reminder to check sales."}] + _listed("feed the calves", "check sales")
    assert answer_issue(Action.fail("I couldn't find a reminder to check sales."), evs) is not None  # it was listed
    other = [{"type": "user", "content": "Delete my reminder to order spare pump parts."}] + _listed("check sales")
    assert answer_issue(Action.fail("I couldn't find a reminder to order spare pump parts."), other) is None
    assert answer_issue(Action.fail("I couldn't find a reminder to order the pump."), other) is not None  # not the user's words
