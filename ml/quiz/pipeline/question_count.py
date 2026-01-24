from __future__ import annotations


def decide_question_count(word_count: int, video_duration_sec: float) -> int:
    """
    Decide number of questions using BOTH:
    - transcript length (word_count)
    - video duration

    Reason:
    - transcripts can be noisy/missing
    - duration is a stable signal
    """

    if word_count <= 0 or video_duration_sec <= 0:
        return 5

    minutes = video_duration_sec / 60.0

    # base from transcript size
    if word_count < 800:
        base = 5
    elif word_count < 1500:
        base = 8
    elif word_count < 2500:
        base = 12
    else:
        base = 15

    # adjust by duration
    if minutes < 6:
        adj = -1
    elif minutes < 12:
        adj = 0
    elif minutes < 20:
        adj = +2
    else:
        adj = +3

    total = base + adj

    # keep safe bounds
    if total < 5:
        total = 5
    if total > 18:
        total = 18

    return total


def split_question_types(total: int) -> dict:
    """
    We will generate:
    - MCQ (includes fill-in-the-blank MCQ)
    - True/False

    Default:
    - 75% MCQ (including blanks)
    - 25% TF
    """
    if total <= 0:
        return {"mcq": 0, "tf": 0}

    mcq = round(total * 0.75)
    tf = total - mcq
    return {"mcq": mcq, "tf": tf}
