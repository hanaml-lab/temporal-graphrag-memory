"""Resolving time expressions against the statement date (never the processing date).

`resolve` maps an expression to the period it denotes, as a half-open [lo, hi) range of day numbers; `since` and
`during` turn periods into the fuzzy valid-time intervals the ledger stores. The rules cover the expressions the
toy setting generates and the most common conversational ones; the paper's pipeline falls back to the extraction
model for anything else.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from .types import FuzzyInterval

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}
NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
           "nine": 9, "ten": 10, "a couple of": 2, "a few": 3, "several": 4}
UNITS = {"day": 1, "week": 7, "month": 30.44, "year": 365.25}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
PREFIX = re.compile(r"^(since|in|from|until|till|during|on|around|about|by)\s+")


def day(d: date) -> float:
    """Day number of a date (proleptic Gregorian ordinal)."""
    return float(d.toordinal())


def _month_start(year: int, month: int) -> date:
    return date(year + (month - 1) // 12, (month - 1) % 12 + 1, 1)


def resolve(expr: str, stmt: date, future: bool = False) -> tuple[float, float] | None:
    text = expr.strip().lower().rstrip(".")
    while PREFIX.match(text):
        text = PREFIX.sub("", text, count=1)
    if text in ("yesterday", "today", "tomorrow"):
        d = stmt + timedelta(days={"yesterday": -1, "today": 0, "tomorrow": 1}[text])
        return day(d), day(d) + 1
    m = re.fullmatch(r"(last|this|next) (week|month|year)", text)
    if m:
        step = {"last": -1, "this": 0, "next": 1}[m.group(1)]
        if m.group(2) == "week":
            monday = stmt - timedelta(days=stmt.weekday()) + timedelta(weeks=step)
            return day(monday), day(monday) + 7
        if m.group(2) == "month":
            lo = _month_start(stmt.year, stmt.month + step)
            return day(lo), day(_month_start(lo.year, lo.month + 1))
        return day(date(stmt.year + step, 1, 1)), day(date(stmt.year + step + 1, 1, 1))
    m = re.fullmatch(r"(\d+|" + "|".join(NUMBERS) + r") (day|week|month|year)s? ago", text)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else NUMBERS[m.group(1)]
        unit = UNITS[m.group(2)]
        centre = day(stmt) - n * unit
        half = max(1.0, unit / 2) if m.group(1) not in ("a few", "several", "a couple of") else n * unit / 2
        return float(round(centre - half)), float(round(centre - half) + round(2 * half))
    m = re.fullmatch(r"next (" + "|".join(WEEKDAYS) + r")", text)
    if m:
        ahead = (WEEKDAYS.index(m.group(1)) - stmt.weekday()) % 7 or 7
        d = stmt + timedelta(days=ahead)
        return day(d), day(d) + 1
    m = re.fullmatch(r"(?:early |late |mid-?)?(" + "|".join(MONTHS) + r")(?: (\d{4}))?", text)
    if m:
        month = MONTHS[m.group(1)]
        if m.group(2):
            year = int(m.group(2))
        elif future:
            year = stmt.year if month > stmt.month else stmt.year + 1
        else:
            year = stmt.year if month <= stmt.month else stmt.year - 1
        lo = date(year, month, 1)
        return day(lo), day(_month_start(year, month + 1))
    m = re.fullmatch(r"(\d{4})", text)
    if m:
        y = int(m.group(1))
        return day(date(y, 1, 1)), day(date(y + 1, 1, 1))
    return None


def since(expr: str, stmt: date) -> FuzzyInterval | None:
    """A state that began in the period `expr` denotes and still holds at the statement date."""
    p = resolve(expr, stmt)
    if p is None:
        return None
    end = day(stmt)
    return FuzzyInterval(min(p[0], end), min(p[1], end), end, end)


def during(expr: str, stmt: date) -> FuzzyInterval | None:
    """A state that held over a stated period: "in 2023", "from 2019 to 2023", "between March and May"."""
    text = expr.strip().lower()
    m = re.fullmatch(r"(?:from|between) (.+?) (?:to|until|and) (.+)", text)
    if m:
        a, b = resolve(m.group(1), stmt), resolve(m.group(2), stmt)
        if a is None or b is None or b[0] < a[1]:
            return None
        return FuzzyInterval(a[0], a[1], b[0], b[1])
    p = resolve(text, stmt)
    return FuzzyInterval.exact(*p) if p else None
