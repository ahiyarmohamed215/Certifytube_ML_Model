from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from verification.quiz.transcript.processor import TranscriptChunk

_STOP_WORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "this",
    "into",
    "your",
    "what",
    "when",
    "where",
    "which",
    "while",
    "then",
    "than",
    "true",
    "false",
}


@dataclass(frozen=True)
class GroundednessResult:
    is_grounded: bool
    reason: str
    matched_chunk_id: str | None = None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _tokens(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]{2,}\b", _normalize(text))


def _keyword_set(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _STOP_WORDS}


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    inter = left_set.intersection(right_set)
    union = left_set.union(right_set)
    return len(inter) / max(1, len(union))


def _best_chunk_for_source(source_segment: str, chunks: List[TranscriptChunk]) -> Tuple[TranscriptChunk | None, float]:
    if not chunks:
        return None, 0.0

    source_tokens = _keyword_set(source_segment)
    best_chunk = None
    best_score = 0.0

    for chunk in chunks:
        score = _jaccard(source_tokens, _keyword_set(chunk.text))
        if _normalize(source_segment) in _normalize(chunk.text):
            score = max(score, 1.0)
        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk, best_score


def validate_question_grounded(
    question: Dict[str, object],
    chunks: List[TranscriptChunk],
    min_overlap: float = 0.22,
) -> GroundednessResult:
    required_fields = ["question", "correct_answer", "explanation", "source_segment"]
    missing = [field for field in required_fields if not question.get(field)]
    if missing:
        return GroundednessResult(False, f"Missing required fields: {missing}")

    source_segment = str(question["source_segment"])
    matched_chunk, source_match_score = _best_chunk_for_source(source_segment, chunks)
    if matched_chunk is None or source_match_score < 0.12:
        return GroundednessResult(False, "Source segment is not traceable to transcript chunks.")

    evidence_text = " ".join(
        [
            str(question.get("question", "")),
            str(question.get("correct_answer", "")),
            str(question.get("explanation", "")),
        ]
    )
    evidence_keywords = _keyword_set(evidence_text)
    chunk_keywords = _keyword_set(matched_chunk.text)

    if not evidence_keywords:
        return GroundednessResult(False, "Insufficient semantic content to validate groundedness.")

    overlap = evidence_keywords.intersection(chunk_keywords)
    capped_denominator = max(1, min(12, len(evidence_keywords)))
    overlap_ratio = len(overlap) / capped_denominator

    answer_text = _normalize(str(question.get("correct_answer", "")))
    answer_directly_present = answer_text and answer_text in _normalize(matched_chunk.text)

    if overlap_ratio < min_overlap and not answer_directly_present:
        return GroundednessResult(
            False,
            (
                "Question appears weakly grounded "
                f"(keyword overlap={overlap_ratio:.2f}, threshold={min_overlap:.2f})."
            ),
            matched_chunk_id=matched_chunk.chunk_id,
        )

    return GroundednessResult(
        True,
        "Grounded against transcript.",
        matched_chunk_id=matched_chunk.chunk_id,
    )
