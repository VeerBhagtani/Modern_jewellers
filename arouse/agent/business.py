"""Deterministic business tools: GST calculation and finding leads in the customer list.

"The model understands, code computes": Arouse copies the amount and rate from the user's
words ("2.5 lakh", "18%") and picks a lead-finding method; this module does the parsing,
the arithmetic (in paise, as integers, so Python and the browser agree exactly) and the
filtering. Mirrored in web/arouse.js.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

from arouse.agent.knowledge import stem, terms

# --- money -------------------------------------------------------------------------------

_AMOUNT = re.compile(r"^(?:₹|rs\.?|inr)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(k|thousand|lakhs?|lacs?|crores?|cr)?\s*(?:rupees|rs\.?|/-)?$",
                     re.I)
_RATE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent|per cent)?$", re.I)
_UNIT = {"k": 1_000, "thousand": 1_000, "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
         "crore": 10_000_000, "crores": 10_000_000, "cr": 10_000_000}


def _decimal_to_int(num: str, scale: int) -> int:
    """'2.5' with scale 100 -> 250, exactly (no binary floating point)."""
    whole, _, frac = num.partition(".")
    digits = len(str(scale)) - 1
    frac = (frac + "0" * digits)[:digits] if digits else ""
    return int(whole or "0") * scale + (int(frac) if frac else 0)


def parse_amount(text: str) -> int | None:
    """Rupee amount as written ("5000", "₹5,000", "Rs. 12,500", "2.5 lakh", "50k") -> paise."""
    m = _AMOUNT.match(text.strip())
    if not m:
        return None
    unit = _UNIT.get((m.group(2) or "").lower(), 1)
    paise = _decimal_to_int(m.group(1).replace(",", ""), 100 * unit) if "." in m.group(1) else \
        int(m.group(1).replace(",", "")) * 100 * unit
    return paise if 0 < paise <= 10**15 else None  # above 10^15 paise JS numbers stop being exact


def parse_rate(text: str) -> int | None:
    """GST rate as written ("18", "18%", "0.25 percent") -> basis points (1800)."""
    m = _RATE.match(text.strip())
    if not m:
        return None
    bp = _decimal_to_int(m.group(1), 100)
    return bp if 0 <= bp <= 10_000 else None


def money(paise: int) -> int | float:
    """Paise -> rupees as a JSON number: whole rupees stay integers (same JSON in Python and JS)."""
    return paise // 100 if paise % 100 == 0 else paise / 100


def fmt_inr(paise: int) -> str:
    """Indian digit grouping: 15254237 paise -> '1,52,542.37'."""
    rupees, p = divmod(paise, 100)
    s = str(rupees)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        s = ",".join([head, *groups, tail]) if head else ",".join([*groups, tail])
    return s + (f".{p:02d}" if p else "")


def fmt_rate(bp: int) -> str:
    whole, frac = divmod(bp, 100)
    return str(whole) if not frac else f"{whole}.{frac:02d}".rstrip("0")


def gst_calculate(amount: str, rate: str, inclusive: bool = False) -> tuple[bool, dict[str, Any]]:
    a, r = parse_amount(amount), parse_rate(rate)
    if a is None:
        return False, {"success": False, "error": f"I couldn't read the amount '{amount}'", "retryable": False}
    if r is None:
        return False, {"success": False, "error": f"I couldn't read the GST rate '{rate}'", "retryable": False}
    if inclusive:
        d = 10_000 + r
        base = (a * 10_000 * 2 + d) // (2 * d)  # round half up
        gst = a - base
    else:
        base = a
        gst = (a * r + 5_000) // 10_000
    cgst = (gst + 1) // 2
    return True, {"success": True, "amount": money(a), "rate": money(r), "inclusive": bool(inclusive),
                  "taxable_value": money(base), "gst": money(gst), "cgst": money(cgst), "sgst": money(gst - cgst),
                  "total": money(base + gst)}


def to_paise(x: int | float) -> int:
    return round(x * 100)


# --- leads -------------------------------------------------------------------------------

CUSTOMER_FILE = "customers.csv"
CUSTOMER_COLUMNS = ["name", "city", "interest", "last_purchase", "total_spent", "birthday", "anniversary"]
METHODS = ("inactive", "occasions", "top", "custom")
INACTIVE_DAYS = 90
OCCASION_DAYS = 30
SHOW = 3
_LEAD_WORDS = """find finding lead leads customer customers people person buyer buyers client clients search look looking
bought buy buys buying purchase purchased like likes liked love loves interest interested idea method use try anyone
everyone someone live lives living based go shop sell sold who""".split()
LEAD_STOPWORDS = frozenset(stem(w) for w in _LEAD_WORDS)


def parse_csv(text: str) -> list[list[str]]:
    """Minimal RFC 4180: commas, double quotes, "" escapes; blank lines skipped."""
    rows, row, field, i, quoted = [], [], [], 0, False
    while i < len(text):
        c = text[i]
        if quoted:
            if c == '"' and text[i + 1:i + 2] == '"':
                field.append('"')
                i += 1
            elif c == '"':
                quoted = False
            else:
                field.append(c)
        elif c == '"':
            quoted = True
        elif c == ",":
            row.append("".join(field).strip())
            field = []
        elif c in "\r\n":
            if c == "\r" and text[i + 1:i + 2] == "\n":
                i += 1
            row.append("".join(field).strip())
            field = []
            if any(row):
                rows.append(row)
            row = []
        else:
            field.append(c)
        i += 1
    row.append("".join(field).strip())
    if any(row):
        rows.append(row)
    return rows


def _iso(s: str) -> date | None:
    try:
        return datetime.strptime(s.strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _month_day(s: str) -> tuple[int, int] | None:
    m = re.search(r"(\d{1,2})-(\d{1,2})$", s.strip())
    if not m:
        return None
    mo, d = int(m.group(1)), int(m.group(2))
    return (mo, d) if 1 <= mo <= 12 and 1 <= d <= 31 else None


def _next_occurrence(md: tuple[int, int], today: date) -> date:
    for year in (today.year, today.year + 1):
        mo, d = md
        while True:  # 29 Feb (or 31st of a short month) falls back to the month's last day
            try:
                occ = date(year, mo, d)
                break
            except ValueError:
                d -= 1
        if occ >= today:
            return occ
    return occ


def _spent(row: dict[str, str]) -> int:
    p = parse_amount(row.get("total_spent", "") or "0")
    return p or 0


def custom_keywords(query: str) -> list[str]:
    return [t for t in terms(query) if t not in LEAD_STOPWORDS]


def find_leads(files: dict[str, str], now: datetime, method: str, query: str | None = None) -> tuple[bool, dict[str, Any]]:
    def err(msg: str) -> tuple[bool, dict[str, Any]]:
        return False, {"success": False, "error": msg, "retryable": False}

    if method not in METHODS:
        return err(f"unknown method {method}")
    if (method == "custom") != (query is not None):
        return err("method custom needs a query; the other methods take none")
    if CUSTOMER_FILE not in files:
        return err(f"file not found: {CUSTOMER_FILE}")
    rows = parse_csv(files[CUSTOMER_FILE])
    if not rows:
        return err(f"{CUSTOMER_FILE} is empty")
    head = [h.strip().lower() for h in rows[0]]
    need = {"inactive": "last_purchase", "occasions": None, "top": "total_spent", "custom": None}[method]
    if "name" not in head or (need and need not in head):
        return err(f"{CUSTOMER_FILE} needs the columns name and {need or 'city'}")
    if method == "occasions" and "birthday" not in head and "anniversary" not in head:
        return err(f"{CUSTOMER_FILE} needs a birthday or anniversary column")
    people = [{h: (r[i] if i < len(r) else "") for i, h in enumerate(head)} for r in rows[1:]]
    today = now.date()
    found: list[tuple[Any, dict[str, str], str]] = []  # (sort key, row, why)
    if method == "inactive":
        for p in people:
            last = _iso(p.get("last_purchase", ""))
            if last and (today - last).days >= INACTIVE_DAYS:
                found.append(((-_spent(p), p["name"]), p, f"last bought {last.isoformat()}"))
    elif method == "occasions":
        for p in people:
            best = None
            for kind in ("birthday", "anniversary"):
                md = _month_day(p.get(kind, ""))
                if md:
                    occ = _next_occurrence(md, today)
                    days = (occ - today).days
                    if days <= OCCASION_DAYS and (best is None or days < best[0]):
                        best = (days, f"{kind} on {occ.isoformat()}")
            if best:
                found.append(((best[0], p["name"]), p, best[1]))
    elif method == "top":
        for p in people:
            if _spent(p) > 0:
                found.append(((-_spent(p), p["name"]), p, f"spent ₹{fmt_inr(_spent(p))}"))
    else:
        keys = custom_keywords(query or "")
        for p in people:
            have = set(terms(" ".join([p.get("name", ""), p.get("city", ""), p.get("interest", "")])))
            if keys and all(k in have for k in keys):
                why = f"likes {p['interest']}" if p.get("interest") else "matches your idea"
                found.append(((-_spent(p), p["name"]), p, why))
    found.sort(key=lambda x: x[0])
    leads = [{"name": p["name"], "city": p.get("city", ""), "why": why} for _, p, why in found[:SHOW]]
    out: dict[str, Any] = {"success": True, "method": method, "count": len(found), "leads": leads}
    if method == "custom":
        out["query"] = query
    return True, out


def sample_customers(now: datetime) -> str:
    """Demo customers.csv with dates relative to `now` (the same rows as the website's sample),
    so every lead-finding method has something to find."""
    def at(n: int) -> date:
        return now.date() + timedelta(days=n)

    def md(n: int, year: int) -> str:
        return f"{year}-{at(n).month:02d}-{at(n).day:02d}"

    rows = [["Asha Mehta", "Pune", "gold necklace", at(-140).isoformat(), "120000", md(12, 1988), ""],
            ["Ravi Shah", "Surat", "diamond ring", at(-20).isoformat(), "45000", "", md(5, 2012)],
            ["Neha Rao", "Pune", "silver anklet", at(-300).isoformat(), "8000", md(150, 1995), ""],
            ["Kavya Iyer", "Jaipur", "gold bangles", at(-95).isoformat(), "76500", md(200, 1991), md(25, 2015)],
            ["Arjun Patel", "Mumbai", "gold coins", at(-10).isoformat(), "210000", md(80, 1979), ""],
            ["Sunita Nair", "Pune", "mangalsutra", at(-200).isoformat(), "54000", "", md(260, 2004)],
            ["Farhan Khan", "Delhi", "wedding rings", at(-45).isoformat(), "98000", md(3, 1993), ""],
            ["Meera Joshi", "Nashik", "earrings", at(-400).isoformat(), "15500", md(330, 1999), ""]]
    return ",".join(CUSTOMER_COLUMNS) + "\n" + "\n".join(",".join(r) for r in rows) + "\n"
