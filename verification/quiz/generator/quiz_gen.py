from __future__ import annotations

import json
import logging
import math
import re
import time
from itertools import cycle
from typing import Dict, List, Literal, Optional

import httpx

from app.core.settings import settings
from verification.quiz.generator.prompts import (
    CODING_PROMPT,
    FILL_BLANK_PROMPT,
    MCQ_PROMPT,
    PLAN_QUIZ_PROMPT,
    SHORT_ANSWER_PROMPT,
    SYSTEM_PROMPT,
    TRUE_FALSE_PROMPT,
)
from verification.quiz.transcript.processor import ProcessedTranscript
from verification.quiz.validator.groundedness import validate_question_grounded

log = logging.getLogger(__name__)

QuestionType = Literal["mcq", "true_false", "fill_blank", "short_answer", "coding"]

PROMPT_MAP = {
    "mcq": MCQ_PROMPT,
    "true_false": TRUE_FALSE_PROMPT,
    "fill_blank": FILL_BLANK_PROMPT,
    "short_answer": SHORT_ANSWER_PROMPT,
    "coding": CODING_PROMPT,
}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_BLOOM = {"remember", "understand", "apply", "analyze"}


class QuizGenerationError(Exception):
    pass


def _ensure_not_timed_out(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise QuizGenerationError("Quiz generation timed out. Reduce question count and try again.")


def _extract_json_object(raw_text: str) -> Dict[str, object]:
    text = (raw_text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise QuizGenerationError("LLM response did not contain a valid JSON object.")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise QuizGenerationError(f"Failed to parse LLM JSON output: {exc}") from exc
    if not isinstance(parsed, dict):
        raise QuizGenerationError("LLM response JSON must be an object.")
    return parsed


def _call_openrouter(prompt: str, request_timeout_seconds: float | None = None) -> Dict[str, object]:
    if not settings.openrouter_api_key:
        raise QuizGenerationError("OPENROUTER_API_KEY is not configured.")
    base_url = settings.openrouter_base_url.rstrip("/")
    if base_url.endswith("/api"):
        base_url = f"{base_url}/v1"
    endpoint = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": settings.openrouter_site_url,
        "X-Title": settings.openrouter_app_name,
    }
    payload = {
        "model": settings.quiz_model,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": prompt},
        ],
    }
    timeout_seconds = settings.llm_timeout_seconds
    if request_timeout_seconds is not None:
        timeout_seconds = min(timeout_seconds, max(1.0, request_timeout_seconds))

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(endpoint, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        log.exception("OpenRouter HTTP error: %s", exc)
        raise QuizGenerationError(f"Failed to contact OpenRouter: {exc}") from exc
    if response.status_code >= 400:
        log.error("OpenRouter returned %d: %s", response.status_code, response.text[:300])
        raise QuizGenerationError(
            f"OpenRouter request failed: {response.status_code} {response.text[:300]}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise QuizGenerationError(
            f"OpenRouter returned non-JSON response: {response.text[:300]}"
        ) from exc
    if not isinstance(data, dict):
        raise QuizGenerationError("OpenRouter response format was unexpected.")
    if "choices" not in data:
        err = data.get("error")
        if isinstance(err, dict):
            msg = str(err.get("message", "")).strip() or str(err)
            raise QuizGenerationError(f"OpenRouter API error: {msg}")
        raise QuizGenerationError(f"OpenRouter response missing choices: {str(data)[:300]}")
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise QuizGenerationError("OpenRouter response format was unexpected.") from exc
    return _extract_json_object(str(content))


def _question_types(count: int, include_coding: bool) -> List[QuestionType]:
    base: List[QuestionType] = ["mcq", "true_false", "fill_blank", "short_answer"]
    if include_coding:
        base.append("coding")
    iterator = cycle(base)
    return [next(iterator) for _ in range(count)]


def _coerce_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group(0))
    return None


def _heuristic_question_count(video_duration_sec: float, max_questions: int) -> int:
    duration_minutes = max(1.0, video_duration_sec / 60.0)
    # Roughly one question per 3 minutes, minimum 4.
    estimated = int(math.ceil(duration_minutes / 3.0))
    return max(4, min(max_questions, estimated))


def _time_budget_question_cap() -> int:
    # Reserve one provider call for planning, then budget remaining time across questions.
    planning_budget = max(6.0, settings.llm_timeout_seconds * 0.9)
    available = max(6.0, settings.quiz_generation_timeout_seconds - planning_budget)
    per_question_budget = max(
        6.0,
        settings.llm_timeout_seconds * 0.9 * max(1, settings.quiz_max_attempts_per_question),
    )
    cap = int(available // per_question_budget)
    return max(1, cap)


def _plan_question_count(
    video_title: str,
    transcript: ProcessedTranscript,
    video_duration_sec: float,
    include_coding: bool,
    max_questions: int,
    deadline: float,
) -> int:
    fallback = _heuristic_question_count(video_duration_sec=video_duration_sec, max_questions=max_questions)
    transcript_excerpt = " ".join(chunk.text for chunk in transcript.chunks[:4]).strip()
    transcript_excerpt = transcript_excerpt[:6000]
    if not transcript_excerpt:
        return fallback

    _ensure_not_timed_out(deadline)
    remaining_budget = deadline - time.monotonic()

    prompt = PLAN_QUIZ_PROMPT.format(
        video_title=video_title,
        video_duration_sec=round(video_duration_sec, 2),
        include_coding=str(include_coding).lower(),
        max_questions=max_questions,
        transcript_excerpt=transcript_excerpt,
    )

    try:
        plan = _call_openrouter(prompt, request_timeout_seconds=remaining_budget)
    except QuizGenerationError as exc:
        log.warning("Planning LLM call failed, using heuristic: %s", exc)
        return fallback

    planned = _coerce_int(plan.get("question_count"))
    if planned is None:
        return fallback
    if planned < 1:
        return 1
    return min(max_questions, planned)


def _default_bloom(question_type: QuestionType) -> str:
    mapping = {
        "mcq": "understand",
        "true_false": "understand",
        "fill_blank": "remember",
        "short_answer": "analyze",
        "coding": "apply",
    }
    return mapping[question_type]


def _sanitize_question(
    raw: Dict[str, object],
    question_type: QuestionType,
    question_id: str,
    source_segment: str,
    difficulty: str,
    bloom_level: str,
) -> Dict[str, object]:
    question_text = str(raw.get("question", "")).strip()
    correct_answer = str(raw.get("correct_answer", "")).strip()
    explanation = str(raw.get("explanation", "")).strip()
    generated_source = str(raw.get("source_segment", "")).strip() or source_segment
    if not question_text or not correct_answer or not explanation:
        raise QuizGenerationError("Generated question is missing mandatory fields.")

    normalized_difficulty = str(raw.get("difficulty", difficulty)).strip().lower()
    if normalized_difficulty not in VALID_DIFFICULTIES:
        normalized_difficulty = difficulty
    normalized_bloom = str(raw.get("bloom_level", bloom_level)).strip().lower()
    if normalized_bloom not in VALID_BLOOM:
        normalized_bloom = bloom_level

    options_value = raw.get("options")
    options: List[str] | None
    if question_type == "mcq":
        if not isinstance(options_value, list):
            raise QuizGenerationError("MCQ must provide 4 options.")
        options = [str(v).strip() for v in options_value if str(v).strip()]
        options = list(dict.fromkeys(options))
        if len(options) != 4 or correct_answer not in options:
            raise QuizGenerationError("MCQ options must be 4 unique values including correct_answer.")
    elif question_type == "true_false":
        value = correct_answer.lower()
        if value in {"true", "t"}:
            correct_answer = "True"
        elif value in {"false", "f"}:
            correct_answer = "False"
        else:
            raise QuizGenerationError("True/False correct_answer must be 'True' or 'False'.")
        options = ["True", "False"]
    else:
        options = None

    return {
        "question_id": question_id,
        "type": question_type,
        "question": question_text,
        "options": options,
        "correct_answer": correct_answer,
        "explanation": explanation,
        "source_segment": generated_source,
        "difficulty": normalized_difficulty,
        "bloom_level": normalized_bloom,
    }


def _build_prompt(
    question_type: QuestionType,
    video_title: str,
    source_segment: str,
    difficulty: str,
    bloom_level: str,
    feedback: str = "",
) -> str:
    prompt = PROMPT_MAP[question_type].format(
        video_title=video_title,
        source_segment=source_segment,
        difficulty=difficulty,
        bloom_level=bloom_level,
    ).strip()
    if feedback:
        prompt = f"{prompt}\n\nValidation feedback: {feedback}\nRegenerate with stronger grounding."
    return prompt


def _generate_one(
    question_type: QuestionType,
    question_id: str,
    video_title: str,
    source_segment: str,
    transcript: ProcessedTranscript,
    difficulty: str,
    bloom_level: str,
    deadline: float,
    max_attempts: int,
) -> Dict[str, object]:
    retry_feedback = ""
    for _ in range(max_attempts):
        _ensure_not_timed_out(deadline)
        remaining_budget = deadline - time.monotonic()
        raw = _call_openrouter(
            _build_prompt(
                question_type=question_type,
                video_title=video_title,
                source_segment=source_segment,
                difficulty=difficulty,
                bloom_level=bloom_level,
                feedback=retry_feedback,
            ),
            request_timeout_seconds=remaining_budget,
        )
        question = _sanitize_question(
            raw=raw,
            question_type=question_type,
            question_id=question_id,
            source_segment=source_segment,
            difficulty=difficulty,
            bloom_level=bloom_level,
        )
        validation = validate_question_grounded(question, transcript.chunks)
        if validation.is_grounded:
            return question
        retry_feedback = validation.reason
    raise QuizGenerationError(f"Question {question_id} failed groundedness checks after {max_attempts} attempts.")


def _fallback_quiz_questions(
    transcript: ProcessedTranscript,
    include_coding: bool,
    count: int,
) -> List[Dict[str, object]]:
    """Generate transcript-grounded fallback questions with varied types."""
    questions: List[Dict[str, object]] = []
    safe_count = max(4, count)  # Always generate at least 4

    # Cycle through all question types for variety
    type_cycle = cycle(["mcq", "true_false", "fill_blank", "short_answer"])
    difficulty_cycle = cycle(["easy", "medium", "hard"])

    for index in range(1, safe_count + 1):
        chunk = transcript.chunks[(index - 1) % len(transcript.chunks)]
        # Take a short excerpt for context in the question
        excerpt = chunk.text[:200].strip()
        q_type = next(type_cycle)
        difficulty = next(difficulty_cycle)

        if q_type == "mcq":
            questions.append(
                {
                    "question_id": f"q{index}",
                    "type": "mcq",
                    "question": f"Based on the video content, which of the following best describes the main concept discussed: '{excerpt}...'?",
                    "options": [
                        "The concept described in this segment is correct as stated",
                        "The concept applies only in unrelated contexts",
                        "The concept contradicts standard practices",
                        "The concept was not mentioned in the video",
                    ],
                    "correct_answer": "The concept described in this segment is correct as stated",
                    "explanation": "This question tests whether you understood the key concept from the video segment.",
                    "source_segment": chunk.text,
                    "difficulty": difficulty,
                    "bloom_level": "understand",
                }
            )
        elif q_type == "true_false":
            questions.append(
                {
                    "question_id": f"q{index}",
                    "type": "true_false",
                    "question": f"True or False: The video segment discusses the following concept — '{excerpt}...'",
                    "options": ["True", "False"],
                    "correct_answer": "True",
                    "explanation": "This statement is directly supported by the video transcript content.",
                    "source_segment": chunk.text,
                    "difficulty": difficulty,
                    "bloom_level": "remember",
                }
            )
        elif q_type == "fill_blank":
            # Extract a key phrase from the excerpt for the blank
            words = excerpt.split()
            if len(words) > 6:
                key_word = words[len(words) // 2]
                blanked = excerpt.replace(key_word, "___", 1)
                questions.append(
                    {
                        "question_id": f"q{index}",
                        "type": "fill_blank",
                        "question": f"Fill in the blank: '{blanked}...'",
                        "options": None,
                        "correct_answer": key_word,
                        "explanation": f"The missing word is '{key_word}', as stated in the video transcript.",
                        "source_segment": chunk.text,
                        "difficulty": difficulty,
                        "bloom_level": "remember",
                    }
                )
            else:
                questions.append(
                    {
                        "question_id": f"q{index}",
                        "type": "short_answer",
                        "question": f"Explain the main idea from this part of the video: '{excerpt}...'",
                        "options": None,
                        "correct_answer": "A strong answer should state the core concept from the segment and a correct usage context.",
                        "explanation": "This checks understanding of the video content.",
                        "source_segment": chunk.text,
                        "difficulty": difficulty,
                        "bloom_level": "understand",
                    }
                )
        else:  # short_answer
            questions.append(
                {
                    "question_id": f"q{index}",
                    "type": "short_answer",
                    "question": f"In your own words, explain what is being discussed in this part of the video: '{excerpt}...'",
                    "options": None,
                    "correct_answer": "A strong answer should state the core concept and demonstrate understanding of the video content.",
                    "explanation": "This checks real understanding of the lesson.",
                    "source_segment": chunk.text,
                    "difficulty": difficulty,
                    "bloom_level": "understand",
                }
            )

    return questions


def generate_quiz(
    session_id: str,
    video_title: str,
    transcript: ProcessedTranscript,
    video_duration_sec: float,
    requested_questions: Optional[int] = None,
    max_questions: int = 20,
    include_coding: bool = False,
) -> List[Dict[str, object]]:
    if not session_id:
        raise QuizGenerationError("session_id is required.")
    if not transcript.chunks:
        raise QuizGenerationError("Transcript could not be chunked. Provide a longer transcript.")
    if video_duration_sec <= 0:
        raise QuizGenerationError("video_duration_sec must be positive.")

    effective_max_questions = min(max_questions, settings.quiz_max_questions)
    if effective_max_questions < 1:
        raise QuizGenerationError("max_questions must be at least 1.")
    if requested_questions is not None and requested_questions > effective_max_questions:
        raise QuizGenerationError(
            f"num_questions exceeds allowed limit ({effective_max_questions})."
        )

    deadline = time.monotonic() + settings.quiz_generation_timeout_seconds
    max_attempts = max(1, int(settings.quiz_max_attempts_per_question))
    planned_count = (
        requested_questions
        if requested_questions is not None
        else _plan_question_count(
            video_title=video_title,
            transcript=transcript,
            video_duration_sec=video_duration_sec,
            include_coding=include_coding,
            max_questions=effective_max_questions,
            deadline=deadline,
        )
    )
    if requested_questions is None:
        planned_count = min(planned_count, _time_budget_question_cap())

    types = _question_types(planned_count, include_coding)
    difficulty_cycle = cycle(["easy", "medium", "hard"])
    questions: List[Dict[str, object]] = []

    for index, question_type in enumerate(types, start=1):
        try:
            _ensure_not_timed_out(deadline)
            chunk = transcript.chunks[(index - 1) % len(transcript.chunks)]
            difficulty = next(difficulty_cycle)
            bloom_level = _default_bloom(question_type)
            questions.append(
                _generate_one(
                    question_type=question_type,
                    question_id=f"q{index}",
                    video_title=video_title,
                    source_segment=chunk.text,
                    transcript=transcript,
                    difficulty=difficulty,
                    bloom_level=bloom_level,
                    deadline=deadline,
                    max_attempts=max_attempts,
                )
            )
        except QuizGenerationError as exc:
            if questions and "timed out" in str(exc).lower():
                break
            if not questions:
                return _fallback_quiz_questions(
                    transcript=transcript,
                    include_coding=include_coding,
                    count=planned_count,
                )
            break

    if not questions:
        return _fallback_quiz_questions(
            transcript=transcript,
            include_coding=include_coding,
            count=planned_count,
        )

    return questions
