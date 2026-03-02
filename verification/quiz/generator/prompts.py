SYSTEM_PROMPT = """
You are an assessment designer for an educational platform.
Create transcript-grounded assessments that evaluate real understanding and skill gain.
Return valid JSON only. No markdown, no prose outside JSON.
"""

PLAN_QUIZ_PROMPT = """
Video title: {video_title}
Video duration (seconds): {video_duration_sec}
Allow coding questions: {include_coding}
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

Generate one true/false misconception check.
Rules:
- Provide statement in "question"
- options must be ["True", "False"]
- correct_answer must be either "True" or "False"

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
- Use exactly one blank represented as "___"
- No options list

Return JSON with keys:
type, question, correct_answer, explanation, source_segment, difficulty, bloom_level
Set type="fill_blank".
"""


SHORT_ANSWER_PROMPT = """
Video title: {video_title}
Difficulty: {difficulty}
Bloom level: {bloom_level}
Transcript segment:
{source_segment}

Generate one short-answer question that checks deeper understanding.
Rules:
- No options list
- correct_answer should be concise but complete
- explanation should justify the expected answer using transcript concepts

Return JSON with keys:
type, question, correct_answer, explanation, source_segment, difficulty, bloom_level
Set type="short_answer".
"""


CODING_PROMPT = """
Video title: {video_title}
Difficulty: {difficulty}
Bloom level: {bloom_level}
Transcript segment:
{source_segment}

Generate one practical coding question from this content.
Rules:
- Ask for a short coding task (not a long project)
- No options list
- correct_answer should include an expected solution approach or snippet
- explanation should explain why that solution is correct

Return JSON with keys:
type, question, correct_answer, explanation, source_segment, difficulty, bloom_level
Set type="coding".
"""
