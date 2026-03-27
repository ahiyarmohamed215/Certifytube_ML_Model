from __future__ import annotations

import re
from typing import Dict, List


class QuizGradingError(Exception):
    pass


def _normalize_answer(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def grade_quiz(
    answer_key: List[Dict[str, str]],
    answers: List[Dict[str, str]],
) -> Dict[str, object]:
    if not answer_key:
        raise QuizGradingError("answer_key must contain at least one question.")
    if not answers:
        raise QuizGradingError("answers must contain at least one submitted answer.")

    key_map = {}
    for item in answer_key:
        question_id = str(item.get("question_id", "")).strip()
        if not question_id:
            raise QuizGradingError("Each answer_key item must include question_id.")
        if question_id in key_map:
            raise QuizGradingError(f"Duplicate question_id in answer_key: {question_id}")
        key_map[question_id] = {
            "correct_answer": str(item.get("correct_answer", "")).strip(),
            "explanation": str(item.get("explanation", "")).strip(),
        }

    answer_map = {}
    for item in answers:
        question_id = str(item.get("question_id", "")).strip()
        if not question_id:
            raise QuizGradingError("Each submitted answer must include question_id.")
        if question_id in answer_map:
            raise QuizGradingError(f"Duplicate question_id in answers: {question_id}")
        answer_map[question_id] = str(item.get("answer", "")).strip()

    unknown_ids = sorted(question_id for question_id in answer_map if question_id not in key_map)
    if unknown_ids:
        raise QuizGradingError(f"Submitted answers contain unknown question_id values: {unknown_ids}")

    results = []
    correct_answers = 0

    for question_id, key_item in key_map.items():
        submitted_answer = answer_map.get(question_id, "")
        is_correct = (
            question_id in answer_map
            and _normalize_answer(submitted_answer) == _normalize_answer(key_item["correct_answer"])
        )
        if is_correct:
            correct_answers += 1

        results.append(
            {
                "question_id": question_id,
                "submitted_answer": submitted_answer,
                "correct_answer": key_item["correct_answer"],
                "explanation": key_item["explanation"],
                "is_correct": is_correct,
            }
        )

    total_questions = len(key_map)
    answered_questions = len(answer_map)
    incorrect_answers = answered_questions - correct_answers
    unanswered_questions = total_questions - answered_questions
    quiz_score_percent = (correct_answers / total_questions) * 100.0 if total_questions else 0.0

    return {
        "total_questions": total_questions,
        "answered_questions": answered_questions,
        "correct_answers": correct_answers,
        "incorrect_answers": incorrect_answers,
        "unanswered_questions": unanswered_questions,
        "quiz_score_percent": round(float(quiz_score_percent), 2),
        "results": results,
    }
