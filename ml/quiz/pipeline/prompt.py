from __future__ import annotations


def build_quiz_prompt(
    video_id: str,
    cleaned_transcript: str,
    mcq_count: int,
    tf_count: int,
) -> str:
    """
    Prompt designed for structured JSON output.
    Rules:
    - Use ONLY the given transcript (no outside knowledge)
    - Every question must be answerable from transcript
    - MCQ must have 4 options when possible
    - Provide answer key + short explanation + source sentence
    """

    return f"""
You are generating a quiz from a YouTube video transcript.

STRICT RULES:
1) Use ONLY the information present in the transcript. Do NOT use external knowledge.
2) Every question must be answerable directly from the transcript.
3) Return JSON only (no markdown, no extra text).
4) For MCQ:
   - Provide 4 choices when possible.
   - Exactly one choice must be correct.
   - Distractors must be plausible but wrong based on transcript.
5) For True/False:
   - Answer must be exactly "True" or "False".
6) For each question:
   - Include a short 1–2 line explanation.
   - Include the exact source sentence from the transcript that supports the answer.
7) Avoid duplicates and avoid trivial questions (e.g., pure greetings or filler lines).
8) Keep stems clear and short. Avoid multi-sentence confusing stems.

TASK:
- Generate {mcq_count} multiple-choice questions (type="mcq")
- Generate {tf_count} true/false questions (type="tf")

VIDEO_ID: {video_id}

TRANSCRIPT (cleaned):
{cleaned_transcript}
""".strip()
