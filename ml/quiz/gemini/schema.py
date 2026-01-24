from __future__ import annotations

"""
This module defines the JSON schema we require Gemini to output.

Why:
- Prevents messy free-text responses
- Makes grading deterministic
- Makes the quiz auditable (questions + answers + evidence)

We will request:
- quiz_id
- video_id
- cleaned_transcript_word_count
- questions[] (MCQ + TF) with answer key + short explanation + source sentence
"""


def quiz_response_schema() -> dict:
    """
    JSON Schema for Gemini structured output.
    Return value is a Python dict that can be passed as response_schema.
    """
    return {
        "type": "object",
        "required": ["quiz_id", "video_id", "cleaned_transcript_word_count", "questions"],
        "properties": {
            "quiz_id": {"type": "string"},
            "video_id": {"type": "string"},
            "cleaned_transcript_word_count": {"type": "integer", "minimum": 0},
            "questions": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["qid", "type", "stem", "answer", "explanation", "source_sentence"],
                    "properties": {
                        "qid": {"type": "string"},
                        "type": {"type": "string", "enum": ["mcq", "tf"]},

                        # Question prompt
                        "stem": {"type": "string", "minLength": 8},

                        # MCQ choices (only required for mcq type)
                        "choices": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 6
                        },

                        # Answer:
                        # - For MCQ: must be one of the choices (exact string)
                        # - For TF: must be "True" or "False"
                        "answer": {"type": "string", "minLength": 1},

                        # Short explanation in plain English (1–2 lines)
                        "explanation": {"type": "string", "minLength": 10},

                        # Evidence line from transcript (auditable)
                        "source_sentence": {"type": "string", "minLength": 10},

                        # Optional difficulty tag (nice for thesis)
                        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    },
                },
            },
        },
        "additionalProperties": False,
    }