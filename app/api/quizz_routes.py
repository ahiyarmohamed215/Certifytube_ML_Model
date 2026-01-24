from __future__ import annotations

import uuid
from fastapi import APIRouter, HTTPException

from app.api.schemas import QuizGenerateRequest, QuizGenerateResponse
from ml.quiz.pipeline.clean import clean_transcript, word_count
from ml.quiz.pipeline.question_count import decide_question_count, split_question_types
from ml.quiz.pipeline.prompt import build_quiz_prompt
from ml.quiz.gemini.generate import generate_quiz_from_transcript

router = APIRouter(prefix="/quiz", tags=["quiz"])

@router.post("/generate", response_model=QuizGenerateResponse)
def generate(req: QuizGenerateRequest):
    def validate_quiz_payload(data: dict):
        if "questions" not in data or not isinstance(data["questions"], list):
            raise ValueError("Gemini output missing 'questions' list")

        for q in data["questions"]:
            qtype = q.get("type")
            if qtype not in ("mcq", "tf"):
                raise ValueError(f"Invalid question type: {qtype}")

            if not q.get("stem") or not q.get("answer"):
                raise ValueError("Question missing stem/answer")

            if qtype == "mcq":
                choices = q.get("choices")
                if not choices or len(choices) != 4:
                    raise ValueError(f"MCQ must have exactly 4 choices: {q.get('qid')}")
                if q.get("answer") not in choices:
                    raise ValueError(f"MCQ answer not in choices: {q.get('qid')}")

            if qtype == "tf":
                if q.get("answer") not in ("True", "False"):
                    raise ValueError(f"TF answer must be True/False: {q.get('qid')}")

            if not q.get("explanation") or not q.get("source_sentence"):
                raise ValueError("Question missing explanation/source_sentence")

        return True

    try:
        cleaned = clean_transcript(req.transcript)
        wc = word_count(cleaned)

        if wc < 120:
            raise HTTPException(status_code=400, detail="Transcript too short to generate a reliable quiz.")

        # Decide number of questions (auto) or override
        if req.num_questions is not None:
            total = req.num_questions
        else:
            total = decide_question_count(wc, req.video_duration_sec)

        split = split_question_types(total)
        mcq_count = split["mcq"]
        tf_count = split["tf"]

        prompt = build_quiz_prompt(
            video_id=req.video_id,
            video_duration_sec=req.video_duration_sec,
            cleaned_transcript=cleaned,
            mcq_count=mcq_count,
            tf_count=tf_count,
        )

        # Attempt 1
        quiz_data = generate_quiz_from_transcript(
            video_id=req.video_id,
            cleaned_transcript=cleaned,
            cleaned_word_count=wc,
            prompt=prompt,
        )

        # Validate; if invalid, retry once (same prompt)
        try:
            validate_quiz_payload(quiz_data)
        except Exception:
            quiz_data = generate_quiz_from_transcript(
                video_id=req.video_id,
                cleaned_transcript=cleaned,
                cleaned_word_count=wc,
                prompt=prompt,
            )
            validate_quiz_payload(quiz_data)

        if not quiz_data.get("quiz_id"):
            quiz_data["quiz_id"] = f"QZ_{uuid.uuid4().hex[:10]}"

        quiz_data["debug"] = {
            "video_duration_sec": req.video_duration_sec,
            "cleaned_word_count": wc,
            "auto_total_questions": total,
            "mcq_count": mcq_count,
            "tf_count": tf_count,
            "difficulty": req.difficulty,
        }

        return quiz_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
