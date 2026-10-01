"""Arouse's reply formats, and the runtime's fallback answer.

The synthetic training data uses these formats for final answers. When the model's own
answer fails the answer guard (it uses words that are not in the conversation, i.e. an
invented fact) and no resample passes, the runtime answers from the last successful tool
result instead, in the same format. The facts then come from the tool, never from the model.
"""

from __future__ import annotations

from typing import Any

from arouse.agent.business import fmt_inr, fmt_rate, to_paise
from arouse.agent.context import DAY_CODES, WEEKDAYS

R_UNKNOWN = "I don't know that yet. I can answer questions about GST, gold and jewellery, dairy and running a small business."
LEADS_ASK = ("I can find leads in 3 ways: 1) past customers who haven't bought in 3 months, 2) customers with a "
             "birthday or anniversary in the next 30 days, 3) your top customers, to ask for referrals. Which one? "
             "Or tell me your own idea.")
LEADS_OWN = "Okay. Tell me your idea: how should I look for leads?"
R_NO_CUSTOMERS = ("I need your customer list to find leads. Add a file named customers.csv with the columns name, city, "
                  "interest, last_purchase, total_spent, birthday and anniversary, then ask me again.")
Q_GST_RATE = "Which GST rate should I use? For example 3% for gold, 5% or 18%."
LEAD_DESC = {"inactive": "no purchase in 3 months", "occasions": "birthday or anniversary in 30 days",
             "top": "top customers, ask them for referrals", "custom": "your idea"}


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def rule_words(repeat: dict[str, Any]) -> str:
    f = repeat["freq"]
    if f == "daily":
        n = repeat.get("interval", 1)
        return "every day" if n == 1 else f"every {n} days"
    if f == "monthly":
        return f"on the {ordinal(repeat['by_month_day'][0])} of every month"
    days = repeat["by_day"]
    if days == list(DAY_CODES[:5]):
        return "every weekday"
    if days == ["SA", "SU"]:
        return "every weekend"
    names = [WEEKDAYS[DAY_CODES.index(d)].capitalize() for d in days]
    return "every " + " and ".join(names)


def created_msg(res: dict[str, Any], repeat: dict[str, Any] | None) -> str:
    date, time = res["next_run"].split("T")
    msg = f"Reminder set: {res['task']} on {date} at {time}"
    return msg + (f", repeating {rule_words(repeat)}." if repeat else ".")


def list_msg(items: list[dict[str, Any]]) -> str:
    if not items:
        return "You have no reminders."
    parts = [f"{x['task']} on {x['next_run'].replace('T', ' at ')}" + (" (repeats)" if "repeat" in x else "") for x in items]
    return f"You have {len(items)} reminder{'s' if len(items) > 1 else ''}: " + "; ".join(parts) + "."


def gst_msg(res: dict[str, Any]) -> str:
    r = fmt_rate(to_paise(res["rate"]))
    a, g, c, sg = (fmt_inr(to_paise(res[k])) for k in ("amount", "gst", "cgst", "sgst"))
    if res.get("inclusive"):
        return f"₹{a} includes ₹{g} GST at {r}% (CGST ₹{c} + SGST ₹{sg}). Price before GST: ₹{fmt_inr(to_paise(res['taxable_value']))}."
    return f"GST at {r}% on ₹{a} is ₹{g} (CGST ₹{c} + SGST ₹{sg}). Total: ₹{fmt_inr(to_paise(res['total']))}."


def leads_msg(res: dict[str, Any]) -> str:
    desc, n, leads = LEAD_DESC[res["method"]], res["count"], res["leads"]
    if n == 0:
        if res["method"] == "custom":
            return f"I didn't find any customers matching \"{res['query']}\" in customers.csv."
        return f"I didn't find any leads ({desc}) in customers.csv."
    items = "; ".join(f"{x['name']} ({x['city']}, {x['why']})" if x["city"] else f"{x['name']} ({x['why']})" for x in leads)
    more = f" Showing {len(leads)}." if n > len(leads) else ""
    return f"Found {n} lead{'s' if n != 1 else ''} ({desc}): {items}.{more}"


def answer_from_result(events: list[dict[str, Any]]) -> str | None:
    """Final answer for the latest successful tool call in this request, or None."""
    for i in range(len(events) - 1, 0, -1):
        ev = events[i]
        if ev["type"] == "user":
            return None
        if ev["type"] == "tool_error":
            return None
        if ev["type"] == "tool_result":
            prev = events[i - 1]
            if prev["type"] != "arouse" or prev["turn"]["action"]["type"] != "tool_call":
                return None
            tool, args, res = prev["turn"]["action"]["tool"], prev["turn"]["action"]["arguments"], ev["content"]
            try:
                if tool == "scheduler.create":
                    return created_msg(res, args.get("repeat"))
                if tool == "scheduler.list":
                    return list_msg(res["reminders"])
                if tool == "scheduler.delete":
                    return f"Deleted the reminder: {res['task']}."
                if tool == "notes.create":
                    return "Saved the note."
                if tool == "file.read":
                    return f"{res['path']} has {res['lines']} lines. It starts with: {res['preview'].split(chr(10))[0]}"
                if tool == "file.list":
                    return f"You have {len(res['files'])} files: {', '.join(res['files'])}."
                if tool == "kb.search":
                    return res["answer"] if res["found"] else R_UNKNOWN
                if tool == "gst.calculate":
                    return gst_msg(res)
                if tool == "leads.find":
                    return leads_msg(res)
            except (KeyError, TypeError, AttributeError, ValueError):
                return None
            return None
    return None


def corrected_confirmation(action: Any, events: list[dict[str, Any]]) -> str | None:
    """If `action` is a finish that confirms a successful tool call but does not state what the
    tool returned, the correct confirmation; else None. ("x has N lines." may leave out the preview.)"""
    if action.type != "finish":
        return None
    expected = answer_from_result(events)
    text = action.result or ""
    if expected is None or text == expected or (expected.startswith(text) and text.endswith(".")):
        return None
    return expected


def corrected_question(action: Any, events: list[dict[str, Any]]) -> str | None:
    """If `action` asks which file to use after a failed file.read and a file.list, the question
    stated from those tool results (same wording as the training data); else None."""
    if action.type != "ask_user" or not (action.question or "").startswith("I couldn't find"):
        return None
    files, path = None, None
    for i in range(len(events) - 1, 0, -1):
        ev = events[i]
        if ev["type"] == "user":
            break
        prev = events[i - 1]
        call = prev["turn"]["action"] if prev["type"] == "arouse" else {}
        if ev["type"] == "tool_result" and call.get("tool") == "file.list" and files is None:
            files = ev["content"].get("files", [])
        if ev["type"] == "tool_error" and call.get("tool") == "file.read" and files is not None:
            path = call["arguments"].get("path")
            break
    if files is None or not path:
        return None
    stem = path.rsplit(".", 1)[0]
    cands = [f for f in files if f.startswith(stem + "_")]
    expected = (f"I couldn't find {path}. Did you mean {cands[0]}?" if cands
                else f"I couldn't find {path}. Which file should I use? Available: {', '.join(files)}.")
    return None if action.question == expected else expected
