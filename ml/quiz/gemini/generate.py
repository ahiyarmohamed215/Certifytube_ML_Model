from __future__ import annotations

import json
from typing import Any, Dict

from google.genai import types

from ml.quiz.gemini.client import get_gemini_client
from ml.quiz.core.settings import quiz_settings
from ml.quiz.gemini.schema import quiz_response_schema


def generate_quiz_from_transcript(
    video_id: str,
    cleaned_transcript: str,
    cleaned_word_count: int,
    prompt: str,
) -> Dict[str, Any]:
    """
    Calls Gemini and returns quiz JSON that matches our schema.
    """

    client = get_gemini_client()

    schema = quiz_response_schema()

    # Structured output config: JSON schema
    generation_config = types.GenerationConfig(
        temperature=0.4,  # lower = less random = more stable quizzes
        response_mime_type="application/json",
        response_schema=schema,
    )

    response = client.models.generate_content(
        model=quiz_settings.gemini_model,
        contents=prompt,
        config=generation_config,
    )

    # The SDK typically returns JSON in response.text
    # Parse it safely:
    try:
        data = json.loads(response.text)
    except Exception as e:
        raise RuntimeError(f"Gemini returned non-JSON or invalid JSON: {e}\nRaw: {response.text}")

    # Add/overwrite metadata fields (ensure consistent)
    data["video_id"] = video_id
    data["cleaned_transcript_word_count"] = int(cleaned_word_count)

    return data
