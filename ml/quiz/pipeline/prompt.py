from __future__ import annotations


def build_quiz_prompt(
    video_id: str,
    video_duration_sec: float,
    cleaned_transcript: str,
    mcq_count: int,
    tf_count: int,
) -> str:
    minutes = round(video_duration_sec / 60.0, 1)

    return f"""
You are generating a quiz from a YouTube video transcript.

STRICT RULES:
1) Use ONLY the transcript content. Do NOT use external knowledge.
2) Every question must be answerable directly from the transcript.
3) Return JSON only (no markdown, no extra text).
4) Question mix:
   - Generate {mcq_count} MCQs total.
   - At least 40% of MCQs must be fill-in-the-blank style (still MCQ).
   - Generate {tf_count} True/False questions.
5) MCQ rules:
   - Provide exactly 4 choices.
   - Exactly one correct choice.
   - Distractors must be plausible but wrong based on transcript.
6) Fill-in-the-blank MCQ rules:
   - Stem must contain a blank like "____".
   - The correct answer must fit the blank.
7) True/False rules:
   - answer must be exactly "True" or "False".
8) For every question include:
   - explanation (1–2 lines)
   - source_sentence copied from transcript that proves the answer
9) Avoid trivial questions (greetings, filler).
10) Avoid duplicates.

VIDEO META:
- VIDEO_ID: {video_id}
- DURATION_MIN: {minutes}

TRANSCRIPT (cleaned):
{cleaned_transcript}
""".strip()
