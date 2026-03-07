SYSTEM_PROMPT = """
You are an assessment designer for an educational platform.
Create transcript-grounded assessments that evaluate real understanding and skill gain.
Return valid JSON only. No markdown, no prose outside JSON.
"""

PLAN_QUIZ_PROMPT = """
Video title: {video_title}
Video duration (seconds): {video_duration_sec}
Maximum allowed questions: {max_questions}
Transcript excerpt:
{transcript_excerpt}

Decide a suitable quiz size to verify understanding and skill from this lesson.
Question count must be between 1 and {max_questions}.

Return JSON with keys:
question_count, rationale
"""


MCQ_PROMPT = """
Video title: {video_title}
Difficulty: {difficulty}
Bloom level: {bloom_level}
Transcript segment:
{source_segment}

Generate one conceptual MCQ.
Rules:
- 4 options exactly
- 1 correct option, 3 plausible distractors
- Test understanding and applied knowledge, not memorization

Return JSON with keys:
type, question, options, correct_answer, explanation, source_segment, difficulty, bloom_level
Set type="mcq".
"""


TRUE_FALSE_PROMPT = """
Video title: {video_title}
Difficulty: {difficulty}
Bloom level: {bloom_level}
Transcript segment:
{source_segment}

Generate one True/False misconception check, but formatted as a 4-option multiple choice question.
Rules:
- 4 options exactly
- 1 correct option, 3 plausible distractors
- Ask the question such that the user has to evaluate the truthfulness of a statement or concept (e.g. "Which of the following statements about X is true?" or "Why is statement Y false?")
- The options should be distinct sentences.

Return JSON with keys:
type, question, options, correct_answer, explanation, source_segment, difficulty, bloom_level
Set type="true_false".
"""


FILL_BLANK_PROMPT = """
Video title: {video_title}
Difficulty: {difficulty}
Bloom level: {bloom_level}
Transcript segment:
{source_segment}

Generate one fill-in-the-blank question for a key concept.
Rules:
- Use exactly one blank represented as "___" in the question text.
- Provide 4 options exactly (4 potential words/phrases that could fit the blank).
- 1 correct option, 3 plausible distractors.

Return JSON with keys:
type, question, options, correct_answer, explanation, source_segment, difficulty, bloom_level
Set type="fill_blank".
"""
