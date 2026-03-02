from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

TOKEN_PER_WORD = 1.3
DEFAULT_MIN_TOKENS = 500
DEFAULT_MAX_TOKENS = 800

_FILLER_WORDS = [
    "uh",
    "um",
    "like",
    "you know",
    "sort of",
    "kind of",
    "basically",
    "actually",
    "literally",
]


@dataclass(frozen=True)
class TranscriptChunk:
    chunk_id: str
    text: str
    token_estimate: int
    has_code: bool


@dataclass(frozen=True)
class ProcessedTranscript:
    chunks: List[TranscriptChunk]
    has_code_content: bool


def estimate_tokens(text: str) -> int:
    words = re.findall(r"\b\w+\b", text)
    return max(1, int(len(words) * TOKEN_PER_WORD))


def looks_like_code(text: str) -> bool:
    code_signals = [
        r"\bdef\b|\bclass\b|\breturn\b|\bimport\b",
        r"\bpublic\b|\bprivate\b|\bstatic\b|\bvoid\b",
        r"\bif\s*\(|\bfor\s*\(|\bwhile\s*\(",
        r"\{|\}|;|=>",
        r"`{3}",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in code_signals)


def clean_transcript(raw_text: str) -> str:
    text = raw_text or ""
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", text)  # remove timestamps
    text = re.sub(r"\[[^\]]+\]", " ", text)  # remove bracketed cues

    filler_pattern = r"\b(" + "|".join(re.escape(word) for word in _FILLER_WORDS) + r")\b"
    text = re.sub(filler_pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_transcript(
    cleaned_text: str,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> List[TranscriptChunk]:
    if not cleaned_text:
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned_text) if s.strip()]
    if not sentences:
        sentences = [cleaned_text]

    min_words = max(1, int(min_tokens / TOKEN_PER_WORD))
    max_words = max(min_words, int(max_tokens / TOKEN_PER_WORD))

    chunks_text: List[str] = []
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        word_count = len(re.findall(r"\b\w+\b", sentence))
        if current and current_words + word_count > max_words:
            chunks_text.append(" ".join(current).strip())
            current = [sentence]
            current_words = word_count
        else:
            current.append(sentence)
            current_words += word_count

    if current:
        chunks_text.append(" ".join(current).strip())

    if len(chunks_text) > 1:
        merged_chunks: List[str] = []
        for text in chunks_text:
            if merged_chunks and len(re.findall(r"\b\w+\b", text)) < min_words:
                merged_chunks[-1] = f"{merged_chunks[-1]} {text}".strip()
            else:
                merged_chunks.append(text)
        chunks_text = merged_chunks

    chunks: List[TranscriptChunk] = []
    for index, text in enumerate(chunks_text, start=1):
        chunks.append(
            TranscriptChunk(
                chunk_id=f"chunk_{index}",
                text=text,
                token_estimate=estimate_tokens(text),
                has_code=looks_like_code(text),
            )
        )
    return chunks


def process_transcript(raw_text: str) -> ProcessedTranscript:
    cleaned = clean_transcript(raw_text)
    chunks = chunk_transcript(cleaned)
    has_code = any(chunk.has_code for chunk in chunks)
    return ProcessedTranscript(chunks=chunks, has_code_content=has_code)
