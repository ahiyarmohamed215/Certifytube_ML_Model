from __future__ import annotations

from typing import List


def _join_behaviors(items: List[str]) -> str:
    # Remove duplicates while preserving order
    seen = set()
    cleaned = []
    for x in items:
        if x and x not in seen:
            cleaned.append(x)
            seen.add(x)

    if not cleaned:
        return "behavioral patterns"

    if len(cleaned) == 1:
        return cleaned[0]

    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"

    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def not_engaged_template(top_negative_behaviors: List[str], top_positive_behaviors: List[str]) -> str:
    neg = _join_behaviors(top_negative_behaviors)
    pos = _join_behaviors(top_positive_behaviors)

    # We want the message to feel evidence-driven, not emotional.
    return (
        f"Rejected because {neg} outweighed {pos}, "
        f"indicating low-quality engagement during the session."
    )


def engaged_template(top_positive_behaviors: List[str], top_negative_behaviors: List[str]) -> str:
    pos = _join_behaviors(top_positive_behaviors)
    neg = _join_behaviors(top_negative_behaviors)

    return (
        f"Accepted because {pos} dominated over {neg}, "
        f"indicating consistent and attentive engagement during the session."
    )
