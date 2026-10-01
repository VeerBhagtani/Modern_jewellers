"""Knowledge base search, GST calculation and lead finding (the deterministic tools)."""

from datetime import datetime

from arouse.agent.answers import LEADS_ASK, gst_msg, leads_msg
from arouse.agent.business import (fmt_inr, find_leads, gst_calculate, parse_amount, parse_csv, parse_rate)
from arouse.agent.knowledge import load, search
from arouse.agent.synth import TEST_KB_QUESTIONS, TEST_UNKNOWN, unknown_question

NOW = datetime(2026, 10, 1, 10, 0)
CUSTOMERS = """name,city,interest,last_purchase,total_spent,birthday,anniversary
Asha Mehta,Pune,gold necklace,2026-05-02,"1,20,000",1990-10-12,
Ravi Shah,Surat,diamond ring,2026-09-20,45000,,2010-10-05
Neha Rao,Pune,silver anklet,2025-12-01,8000,1995-03-01,
"Iyer, Kavya",Delhi,gold chain,2026-01-15,30000,2000-02-29,"""


def test_every_entry_is_found_by_its_own_phrasings_and_answers_are_short():
    for e in load()["entries"]:
        for q in [e["q"], *e["alts"]]:
            assert search(q).get("id") == e["id"], q
        assert len(e["a"]) <= 260 and e["source"] in load()["kb"]["sources"]


def test_unknown_questions_are_not_answered():
    import random

    rng = random.Random(3)
    qs = TEST_UNKNOWN + [unknown_question(rng) for _ in range(300)]
    hits = [q for q in qs if search(q)["found"]]
    assert len(hits) <= len(qs) * 0.02, hits[:10]


def test_held_out_questions_mostly_find_the_right_entry():
    right = sum(search(q).get("id") == eid for eid, q in TEST_KB_QUESTIONS)
    assert right >= len(TEST_KB_QUESTIONS) * 0.8  # measured once: 27/33; never tuned on these


def test_amounts_and_rates_are_parsed_exactly():
    cases = {"5000": 500000, "5,000": 500000, "₹5,000": 500000, "Rs. 12,500": 1250000, "rs 750": 75000, "INR 99": 9900,
             "2.5 lakh": 25000000, "1 lakh": 10000000, "1.2 crore": 1200000000, "50k": 5000000, "75 thousand": 7500000,
             "5000 rupees": 500000, "1,25,000": 12500000, "12.75": 1275}
    for text, paise in cases.items():
        assert parse_amount(text) == paise, text
    assert parse_amount("lots") is None and parse_amount("0") is None and parse_amount("-5") is None
    assert parse_rate("18") == 1800 and parse_rate("18%") == 1800 and parse_rate("0.25 percent") == 25
    assert parse_rate("abc") is None and parse_rate("150") is None


def test_gst_calculation_and_reply():
    ok, r = gst_calculate("2.5 lakh", "18%")
    assert ok and (r["gst"], r["cgst"], r["sgst"], r["total"]) == (45000, 22500, 22500, 295000)
    assert gst_msg(r) == "GST at 18% on ₹2,50,000 is ₹45,000 (CGST ₹22,500 + SGST ₹22,500). Total: ₹2,95,000."
    ok, r = gst_calculate("₹1,180", "18", inclusive=True)
    assert ok and (r["taxable_value"], r["gst"]) == (1000, 180)
    assert gst_msg(r) == "₹1,180 includes ₹180 GST at 18% (CGST ₹90 + SGST ₹90). Price before GST: ₹1,000."
    ok, r = gst_calculate("Rs. 12,345", "0.25")
    assert ok and r["gst"] == 30.86 and r["cgst"] + r["sgst"] == 30.86 and r["total"] == 12375.86
    assert gst_calculate("lots", "18")[0] is False and gst_calculate("100", "x")[0] is False
    assert fmt_inr(15254237) == "1,52,542.37" and fmt_inr(100) == "1" and fmt_inr(123456700) == "12,34,567"


def test_csv_parsing_handles_quotes():
    rows = parse_csv(CUSTOMERS)
    assert rows[1][4] == "1,20,000" and rows[4][0] == "Iyer, Kavya" and len(rows) == 5


def test_lead_methods():
    files = {"customers.csv": CUSTOMERS}
    ok, r = find_leads(files, NOW, "inactive")
    assert ok and [x["name"] for x in r["leads"]] == ["Asha Mehta", "Iyer, Kavya", "Neha Rao"]  # by spend
    ok, r = find_leads(files, NOW, "occasions")
    assert [x["why"] for x in r["leads"]] == ["anniversary on 2026-10-05", "birthday on 2026-10-12"]
    ok, r = find_leads(files, NOW, "top")
    assert r["count"] == 4 and len(r["leads"]) == 3 and r["leads"][0]["why"] == "spent ₹1,20,000"
    ok, r = find_leads(files, NOW, "custom", "customers in Pune who like gold")
    assert r["count"] == 1 and r["leads"][0]["name"] == "Asha Mehta"
    assert leads_msg(r) == "Found 1 lead (your idea): Asha Mehta (Pune, likes gold necklace)."
    ok, r = find_leads(files, NOW, "custom", "people from Thane")
    assert ok and r["count"] == 0 and "didn't find" in leads_msg(r)
    assert find_leads({}, NOW, "top") == (False, {"success": False, "error": "file not found: customers.csv", "retryable": False})
    assert not find_leads(files, NOW, "custom")[0] and not find_leads(files, NOW, "top", "x")[0]


def test_leap_day_birthday_and_three_methods_offered():
    ok, r = find_leads({"customers.csv": CUSTOMERS}, datetime(2027, 2, 20, 9, 0), "occasions")
    assert "birthday on 2027-02-28" in [x["why"] for x in r["leads"]]  # 29 Feb in a non-leap year
    assert "1)" in LEADS_ASK and "2)" in LEADS_ASK and "3)" in LEADS_ASK and "own idea" in LEADS_ASK
