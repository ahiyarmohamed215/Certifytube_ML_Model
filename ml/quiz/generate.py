"""
Gemini-based quiz generator.
Sends a YouTube video URL to Gemini and returns structured quiz JSON.
"""
from __future__ import annotations

import json
import uuid

from google import genai
from google.genai import types

from app.core.settings import settings, quiz_settings
from ml.quiz.prompt import build_quiz_prompt


def generate_quiz(
    video_id: str,
    difficulty: str = "medium",
    num_questions: int = 5,
) -> dict:
    """
    Generate a quiz from a YouTube video using Gemini.

    Args:
        video_id:       YouTube video ID (e.g. "dQw4w9WgXcQ").
        difficulty:     "easy", "medium", or "hard".
        num_questions:  Number of questions to generate (1-20).

    Returns:
        A dict matching the QuizGenerateResponse schema.

    Raises:
        ValueError:  If Gemini returns unparseable output.
        Exception:   On API errors.
    """
    # 1) Build the YouTube URL
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"

    # 2) Build the prompt
    prompt = build_quiz_prompt(
        video_id=video_id,
        difficulty=difficulty,
        num_questions=num_questions,
    )

    # 3) Create genai client
    client = genai.Client(api_key=settings.gemini_api_key)

    # 4) Call Gemini with the YouTube video + prompt
    response = client.models.generate_content(
        model=quiz_settings.gemini_model,
        contents=types.Content(
            parts=[
                types.Part(
                    file_data=types.FileData(file_uri=youtube_url)
                ),
                types.Part(text=prompt),
            ]
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    # 5) Parse the JSON response
    raw_text = response.text
    if not raw_text:
        raise ValueError("Gemini returned an empty response.")

    try:
        quiz_data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned invalid JSON: {e}\nRaw: {raw_text[:500]}")

    # 6) Ensure quiz_id is set
    if not quiz_data.get("quiz_id"):
        quiz_data["quiz_id"] = f"QZ_{uuid.uuid4().hex[:10]}"

    # 7) Ensure video_id is set
    quiz_data["video_id"] = video_id

    return quiz_data
