from __future__ import annotations

import re


def clean_transcript(raw: str) -> str:
    """
    Clean transcript text received from backend.

    Removes:
    - timestamps like 00:01 or 00:01:23
    - bracket noise like [Music], (applause)
    - extra whitespace

    Keeps:
    - normal sentences (best for question generation)
    """
    if not raw:
        return ""

    text = raw

    # Remove timestamps e.g., 00:12 or 01:02:33
    text = re.sub(r"\b\d{1,2}:\d{2}(:\d{2})?\b", " ", text)

    # Remove bracketed noise like [Music], [Applause], (laughs)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\([^\)]+\)", " ", text)

    # Remove repeated dashes/underscores etc.
    text = re.sub(r"[_\-]{2,}", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())
