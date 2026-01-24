from __future__ import annotations


def decide_question_count(word_count: int) -> int:
    """
    Decide number of questions based on transcript length (word count).

    Rationale (simple + defendable):
    - short transcripts shouldn't generate too many questions (low quality)
    - long transcripts can support more questions, but we cap it

    Returns: total questions (MCQ + True/False combined)
    """
    if word_count <= 0:
        return 0

    if word_count < 800:
        return 5
    if word_count < 1500:
        return 8
    if word_count < 2500:
        return 12
    return 15  # cap


def split_question_types(total: int) -> dict:
    """
    Split questions into MCQ and True/False.
    Default: 70% MCQ, 30% True/False
    """
    if total <= 0:
        return {"mcq": 0, "tf": 0}

    mcq = round(total * 0.7)
    tf = total - mcq
    return {"mcq": mcq, "tf": tf}
