from __future__ import annotations

from typing import List, Set


def _dedupe(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x and x not in seen and x != "other":
            out.append(x)
            seen.add(x)
    return out


def _join(items: List[str]) -> str:
    if not items:
        return "key behavioural signals"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_technical_explanation(status: str, pos_behaviors: List[str], neg_behaviors: List[str]) -> str:
    pos = _dedupe(pos_behaviors)
    neg = _dedupe(neg_behaviors)

    # avoid duplicates like "coverage vs coverage"
    neg = [b for b in neg if b not in set(pos)]

    if status == "ENGAGED":
        return f"Accepted: positive signals ({_join(pos)}) outweighed negative signals ({_join(neg)})."
    return f"Rejected: negative signals ({_join(neg)}) outweighed positive signals ({_join(pos)})."
