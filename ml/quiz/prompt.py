"""
Quiz prompt builder for Gemini video-based quiz generation.
"""
from __future__ import annotations


def build_quiz_prompt(
    video_id: str,
    difficulty: str = "medium",
    num_questions: int = 5,
) -> str:
    """
    Build a structured prompt that tells Gemini to generate quiz questions
    from the YouTube video it just watched.

    Args:
        video_id:       YouTube video ID (for context in the prompt).
        difficulty:     "easy", "medium", or "hard".
        num_questions:  Total number of questions to generate.

    Returns:
        The prompt string.
    """
    # Rough split: ~60 % MCQ, ~40 % True/False (at least 1 of each when possible)
    mcq_count = max(1, round(num_questions * 0.6))
    tf_count = max(1, num_questions - mcq_count)
    # Adjust if rounding pushed the total over
    if mcq_count + tf_count > num_questions:
        tf_count = num_questions - mcq_count

    return f"""You are a quiz generator for an educational platform called CertifyTube.

Based on the YouTube video you just watched (video ID: {video_id}), generate a quiz with exactly {num_questions} questions.

Requirements:
- {mcq_count} multiple-choice questions (MCQ) with exactly 4 choices each.
- {tf_count} true/false questions (TF).
- Difficulty level: {difficulty}.
- Each question MUST be directly answerable from the video content.
- Provide a clear, concise explanation for each answer referencing the video.

Return your response as a JSON object with this exact structure:
{{
  "quiz_id": "",
  "video_id": "{video_id}",
  "questions": [
    {{
      "qid": "Q1",
      "type": "mcq",
      "stem": "What is ...?",
      "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "answer": "A) ...",
      "explanation": "According to the video, ...",
      "difficulty": "{difficulty}"
    }},
    {{
      "qid": "Q2",
      "type": "tf",
      "stem": "True or False: ...",
      "choices": null,
      "answer": "True",
      "explanation": "The video states that ...",
      "difficulty": "{difficulty}"
    }}
  ]
}}

Rules:
- "qid" must be sequential: Q1, Q2, Q3, ...
- "type" must be either "mcq" or "tf".
- For MCQ, "choices" is a list of 4 strings; "answer" must be one of them.
- For TF, "choices" must be null; "answer" must be "True" or "False".
- Return ONLY the JSON object, no markdown fences, no extra text.
"""
