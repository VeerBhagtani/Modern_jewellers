"""Scheduling intelligence in the training/benchmark data: every gold action is correct."""

from datetime import datetime, timedelta

import pytest

from arouse.agent.context import build_context, next_weekday_date
from arouse.agent.episode import turn_from_event
from arouse.agent.synth import Q_TIME, Q_WHEN, Generator, episode_sandbox, generate, rule_words, say_time

EPISODES = generate(1500, seed=5) + generate(300, seed=6, split="test")


def test_context_calendar():
    ctx = build_context(datetime(2026, 9, 30, 19, 21))
    assert ctx["weekday"] == "wednesday" and ctx["tomorrow"] == "2026-10-01"
    assert ctx["next"]["mon"] == "2026-10-05" and ctx["next"]["wed"] == "2026-10-07"  # "Wednesday" = next week
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
        ctx = build_context(datetime.strptime(ep["now"], "%Y-%m-%dT%H:%M"))
        # the most recent message of this request that names a day decides the date
        norm = [f" {u} ".replace(",", " ").replace(".", " ").replace("?", " ") for u in users]
        day_msgs = [u for u in norm if "tomorrow" in u or any(f" {d} " in u for d in DAY_NAMES.values())]
        if "date" in args and day_msgs and "every" not in day_msgs[-1].lower():
            u = day_msgs[-1]
            if "tomorrow" in u and "today" not in u:
                assert args["date"] == ctx["tomorrow"] or "passed" in str(ep), u
            for w, full in DAY_NAMES.items():
                if f" {full} " in u:
                    assert args["date"] == ctx["next"][w], u
        if "in_minutes" in args:
            assert any(k in users[-1].lower() for k in ("in ", "after", "from now"))
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
