"""Deterministic totalling of pasted market / shopping / expense lists.

Small local models guess at messy lists ("Meat- 40k 8 kg") and are poor at arithmetic.
When someone pastes a list and asks for the total, this parses it with plain rules and adds
it up exactly, so the answer does not depend on the model.

Rule of thumb used: the amount written next to an item is the TOTAL for that line.
Sizes and quantities ("8 kg", "4 packs", "one rubber") are descriptions. A line is only
multiplied when it says so explicitly ("4 x 750", "750 each").
"""
import re
from decimal import Decimal

import config
from services.calculator import _plain, _pretty

ASK = re.compile(r"\b(total|sum|add(?:ed)?\s*(?:it\s*)?up|adding|calculate|how much|worth)\b", re.I)
INSTRUCTION = re.compile(
    r"\b(calculate|total|sum|add(?:ed)?\s*(?:it\s*)?up|how much|please|work out|the list|for me|worth)\b",
    re.I,
)
CURRENCY = re.compile(r"[₦$€£]")

# number, optionally with thousands separators, optionally followed directly by k (= thousand)
NUMBER = re.compile(r"(?<![\w.,])(?:[₦$€£]\s?)?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(k\b)?", re.I)
TIMES = re.compile(
    r"(?<![\w.,])(\d+(?:\.\d+)?)\s*(?:x|×|\*|@)\s*(?:[₦$€£]\s?)?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(k\b)?",
    re.I,
)
EACH = re.compile(r"\b(each|ea|per\s+\w+)\b|/\s*(?:kg|pack|piece|pc|unit|bag|carton|dozen)\b", re.I)
SEPARATOR = re.compile(r"\s*[-–—:=](?=[\s₦$€£\d(])\s*|\s+(?=[₦$€£]?\d)|\s*\(")


def _value(number, k):
    value = Decimal(number.replace(",", ""))
    return value * 1000 if k else value


def _is_price(number, k):
    return bool(k) or _value(number, None) >= 100


def _clean(text):
    text = re.sub(r"[()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -–—:=,.;")
    return (text[:67] + "...") if len(text) > 70 else text


def parse_line(line):
    """Return {"name", "details", "amount" (Decimal|None), "multiplied"} or None to skip the line."""
    core = line.strip()
    if not core:
        return None
    if core.startswith("(") and core.endswith(")"):
        core = core[1:-1].strip()

    # Prefer the part before any "(...)" note, so store names in brackets never matter.
    main = core.split("(")[0]
    amount, matched, multiplied = None, None, False

    for text in (main, core):
        times = TIMES.search(text)
        if times and _is_price(times.group(2), times.group(3)):
            amount = _value(times.group(1), None) * _value(times.group(2), times.group(3))
            matched, multiplied = times.group(0), True
            break

        candidates = [m for m in NUMBER.finditer(text) if _is_price(m.group(1), m.group(2))]
        if candidates:
            m = candidates[0]
            amount, matched = _value(m.group(1), m.group(2)), m.group(0)
            if EACH.search(text[m.end():m.end() + 25]):
                qty = next(
                    (
                        _value(n.group(1), None)
                        for n in NUMBER.finditer(text)
                        if n is not m and not n.group(2) and _value(n.group(1), None) < 100
                    ),
                    None,
                )
                if qty:
                    amount, multiplied = amount * qty, True
            break

    if amount is None and INSTRUCTION.search(core):
        return None                       # "calculate the total", "please add it up" ...
    if amount is None and core.endswith(":") and not re.search(r"\d", core):
        return None                       # a heading such as "Market list:"

    name = SEPARATOR.split(core, maxsplit=1)[0].strip(" -–—:=,.;()") or "Item"
    rest = core[len(SEPARATOR.split(core, maxsplit=1)[0]):]
    if matched:
        rest = rest.replace(matched, " ", 1)
    return {
        "name": name,
        "details": _clean(rest),
        "amount": amount,
        "multiplied": multiplied,
    }


def try_total_list(message):
    """If `message` is a priced list plus a request for the total, return the finished answer."""
    if "```" in message or not ASK.search(message):
        return None

    rows = [r for r in (parse_line(l) for l in message.splitlines()) if r]
    priced = [r for r in rows if r["amount"] is not None]

    if len(priced) < 3 or len(priced) / len(rows) < 0.5:
        return None

    total = sum((r["amount"] for r in priced), Decimal(0))
    unpriced = [r["name"] for r in rows if r["amount"] is None]
    found = CURRENCY.search(message)
    symbol = found.group(0) if found else config.CURRENCY

    def money(value):
        return f"{symbol}{_pretty(value)}"

    lines = [
        "Here is your list, added up exactly:",
        "",
        "| # | Item | Details | Amount |",
        "|---|---|---|---|",
    ]
    for i, r in enumerate(priced, start=1):
        lines.append(f"| {i} | {r['name']} | {r['details'] or ''} | {money(r['amount'])} |")

    lines += ["", f"**Grand total: {money(total)}** ({len(priced)} priced items)"]

    if unpriced:
        lines += ["", f"**No price given, so not counted:** {', '.join(unpriced)}."]

    multiplied = [r["name"] for r in priced if r["multiplied"]]
    lines += [
        "",
        "*How I read it: the amount on each line is the total for that item "
        '(so "Meat 40k, 8 kg" counts as 40,000, not 8 × 40,000), and k means thousand.'
        + (f" I multiplied these because they said so: {', '.join(multiplied)}." if multiplied else "")
        + " If any amount is actually per kg or per pack, tell me which and I will recalculate.*",
    ]

    return {"text": "\n".join(lines), "total": _plain(total), "items": len(priced)}
