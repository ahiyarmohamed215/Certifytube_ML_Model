from verification.quiz.generator.quiz_gen import _question_types, _sanitize_question


def test_question_type_distribution_without_coding():
    question_types = _question_types(6, include_coding=False)
    assert question_types == ["mcq", "true_false", "fill_blank", "short_answer", "mcq", "true_false"]


def test_question_type_distribution_with_coding():
    question_types = _question_types(7, include_coding=True)
    assert question_types == [
        "mcq",
        "true_false",
        "fill_blank",
        "short_answer",
        "coding",
        "mcq",
        "true_false",
    ]


def test_sanitize_mcq_question():
    raw_question = {
        "question": "Which metric measures precision-recall area?",
        "options": ["AUC-PR", "MSE", "RMSE", "MAE"],
        "correct_answer": "AUC-PR",
        "explanation": "AUC-PR summarizes precision-recall behavior.",
        "source_segment": "The transcript explains AUC-PR is useful for class imbalance.",
        "difficulty": "medium",
        "bloom_level": "understand",
    }

    sanitized = _sanitize_question(
        raw=raw_question,
        question_type="mcq",
        question_id="q1",
        source_segment=raw_question["source_segment"],
        difficulty="medium",
        bloom_level="understand",
    )

    assert sanitized["question_id"] == "q1"
    assert sanitized["type"] == "mcq"
    assert len(sanitized["options"]) == 4
    assert sanitized["correct_answer"] in sanitized["options"]
