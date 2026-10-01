"""Scheduling intelligence in the training/benchmark data: every gold action is correct."""

from datetime import datetime, timedelta

import pytest

from arouse.agent.context import build_context, next_weekday_date, upcoming
from arouse.agent.episode import turn_from_event
from arouse.agent.synth import Q_TIME, Q_WHEN, Generator, episode_sandbox, generate, rule_words, say_time
from tests.conftest import ROOT as ROOT_DIR

EPISODES = generate(1500, seed=5) + generate(300, seed=6, split="test")


def test_context_calendar():
    ctx = build_context(datetime(2026, 9, 30, 19, 21))
    assert ctx["calendar"].startswith("today 2026-09-30 (Wednesday), tomorrow 2026-10-01, Thursday 2026-10-01")
    assert ctx["calendar"].endswith("Wednesday 2026-10-07")  # "on Wednesday" said on a Wednesday = next week
    assert ctx["calendar"].count(" Wednesday") == 1  # today's weekday only in parentheses
    u = upcoming(datetime(2026, 9, 30, 19, 21))
    assert u["tomorrow"] == "2026-10-01" and u["monday"] == "2026-10-05" and u["wednesday"] == "2026-10-07"
    assert next_weekday_date(datetime(2026, 9, 30), 3).strftime("%Y-%m-%d") == "2026-10-01"


def test_rule_words():
    assert rule_words({"freq": "weekly", "by_day": ["MO"]}) == "every Monday"
    assert rule_words({"freq": "weekly", "by_day": ["MO", "TH"]}) == "every Monday and Thursday"
    assert rule_words({"freq": "weekly", "by_day": ["MO", "TU", "WE", "TH", "FR"]}) == "every weekday"
    assert rule_words({"freq": "daily", "interval": 3}) == "every 3 days"
    assert rule_words({"freq": "monthly", "by_month_day": [1]}) == "on the 1st of every month"


def test_say_time_forms_are_parseable():
    g = Generator(0)
    forms = {say_time(g.rng, 20, 30) for _ in range(200)}
    assert "20:30" in forms and "8:30 PM" in forms and "8:30 in the evening" in forms
    assert say_time(g.rng, 7, 0) in {"7 AM", "7 am", "7am", "07:00", "7 in the morning"}


def _creates():
    """(episode, user messages of the current request, create arguments)."""
    for ep in EPISODES:
        users: list[str] = []
        done = True
        for ev in ep["events"]:
            if ev["type"] == "user":
                users = [ev["content"]] if done else users + [ev["content"]]
                done = False
            if ev["type"] == "arouse":
                a = turn_from_event(ev).action
                if a.type == "tool_call" and a.tool == "scheduler.create":
                    yield ep, users, a.arguments
                if a.type in ("finish", "fail"):
                    done = True


DAY_NAMES = {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday", "fri": "Friday",
             "sat": "Saturday", "sun": "Sunday"}


def test_every_gold_create_is_valid_and_consistent():
    n = 0
    for ep, users, args in _creates():
        n += 1
        ctx = upcoming(datetime.strptime(ep["now"], "%Y-%m-%dT%H:%M"))
        # the latest real request (not a bare answer like "7:30 pm" or "yes") decides the date
        norm = [f" {u} ".replace(",", " ").replace(".", " ").replace("?", " ") for u in users]
        requests = [u for u in norm if len(u.split()) > 4]
        u = requests[-1] if requests else norm[-1]
        if "date" in args and "every" not in u.lower():
            if "tomorrow" in u and "today" not in u:
                assert args["date"] == ctx["tomorrow"] or "passed" in str(ep), u
            for w, full in DAY_NAMES.items():
                if f" {full} " in u:
                    assert args["date"] == ctx[full.lower()], u
        if "in_minutes" in args:
            assert any(k in users[-1].lower() for k in ("in ", "after", "from now", "timer for"))
        if "repeat" in args:
            assert "date" not in args and "time" in args
    assert n > 300


def test_ambiguous_requests_ask_instead_of_guessing():
    asked = 0
    for ep in EPISODES:
        evs = ep["events"]
        for i, ev in enumerate(evs):
            if ev["type"] != "arouse":
                continue
            a = turn_from_event(ev).action
            if a.type == "ask_user" and a.question in (Q_TIME, Q_WHEN):
                asked += 1
                assert evs[i - 1]["type"] == "user"  # asked immediately, before any tool call
    assert asked > 100


def test_gold_trajectories_replay_exactly():
    """Re-executing each gold tool call reproduces the recorded observation."""
    for ep in EPISODES[:600]:
        sb = episode_sandbox(ep)
        evs = ep["events"]
        for i, ev in enumerate(evs):
            if ev["type"] == "arouse":
                a = turn_from_event(ev).action
                if a.type == "tool_call":
                    ok, payload = sb.execute(a.tool, a.arguments)
                    assert evs[i + 1] == {"type": "tool_result" if ok else "tool_error", "content": payload}
        assert sb.snapshot() == ep["final_state"]


@pytest.mark.parametrize("split", ["train", "test"])
def test_generation_is_deterministic(split):
    assert generate(20, seed=3, split=split) == generate(20, seed=3, split=split)


def test_one_time_reminders_are_in_the_future():
    for ep, _users, args in _creates():
        if "date" in args:
            now = datetime.strptime(ep["now"], "%Y-%m-%dT%H:%M")
            when = datetime.strptime(f"{args['date']}T{args['time']}", "%Y-%m-%dT%H:%M")
            # the only past times in gold data are "today" requests that the scheduler rejects
            assert when > now or when.date() == now.date()
            assert when - now < timedelta(days=80)


def test_every_user_message_gets_a_reply_and_test_split_is_frozen():
    import hashlib
    import json

    for ep in generate(1500, seed=41):
        evs = ep["events"]
        for i, ev in enumerate(evs):
            if ev["type"] == "user":
                assert i + 1 < len(evs) and evs[i + 1]["type"] == "arouse", ep["id"]
    # the held-out benchmark must stay byte-identical when the train generator changes
    # (AgentBench v1 was made by the v1-v3 generator and is frozen on disk; MANIFEST sha256s guard it)
    test = generate(480, 9002, "test")
    blob = "".join(json.dumps(e, ensure_ascii=False, separators=(",", ":")) + "\n" for e in test).encode()
    committed = (ROOT_DIR / "benchmarks/agentbench_v2/test.jsonl").read_bytes()
    assert hashlib.sha256(blob).hexdigest() == hashlib.sha256(committed).hexdigest()


def test_held_out_questions_leads_and_gst_phrasings_are_unseen_in_training():
    from arouse.agent.synth import (TEST_CITIES, TEST_DIRECT, TEST_INTERESTS, TEST_KB_QUESTIONS, TEST_LEAD_ASKS,
                                    TEST_PICKS, TEST_REJECTS, TEST_UNKNOWN)

    train_users = {ev["content"].lower().rstrip("?.!") for ep in generate(4000, seed=104)
                   for ev in ep["events"] if ev["type"] == "user"}
    held_out = [q for _, q in TEST_KB_QUESTIONS] + TEST_UNKNOWN + TEST_LEAD_ASKS + TEST_REJECTS
    held_out += [x for v in TEST_PICKS.values() for x in v] + [x for v in TEST_DIRECT.values() for x in v]
    assert not {h.lower().rstrip("?.!") for h in held_out} & train_users
    joined = " ".join(train_users)
    assert not any(c.lower() in joined for c in TEST_CITIES + TEST_INTERESTS)
