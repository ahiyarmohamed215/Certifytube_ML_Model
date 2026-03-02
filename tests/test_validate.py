from verification.quiz.transcript.processor import chunk_transcript, process_transcript
from verification.quiz.validator.groundedness import validate_question_grounded


def test_process_transcript_detects_code():
    transcript = (
        "Today we define a function in Python. "
        "def add(a, b): return a + b. "
        "This returns the sum of two values."
    )
    processed = process_transcript(transcript)
    assert processed.chunks
    assert processed.has_code_content is True


def test_chunking_creates_multiple_chunks_for_long_text():
    sentence = "Machine learning models learn from feature patterns and improve with feedback. "
    long_text = sentence * 900
    chunks = chunk_transcript(long_text)
    assert len(chunks) >= 2
    assert all(chunk.token_estimate > 0 for chunk in chunks)


def test_groundedness_accepts_grounded_question():
    processed = process_transcript(
        "Gradient descent updates model weights by moving opposite to the gradient. "
        "A learning rate controls the update step size."
    )
    question = {
        "question": "What controls the size of each gradient descent update?",
        "correct_answer": "The learning rate controls the update step size.",
        "explanation": "The transcript states that learning rate controls update step size.",
        "source_segment": processed.chunks[0].text,
    }

    result = validate_question_grounded(question, processed.chunks)
    assert result.is_grounded is True


def test_groundedness_rejects_hallucinated_question():
    processed = process_transcript(
        "Decision trees split data by feature thresholds to reduce impurity in each branch."
    )
    question = {
        "question": "Which GPU architecture is required for transformer pretraining?",
        "correct_answer": "NVIDIA Hopper",
        "explanation": "The lesson focused on GPU architecture choices.",
        "source_segment": processed.chunks[0].text,
    }

    result = validate_question_grounded(question, processed.chunks)
    assert result.is_grounded is False
