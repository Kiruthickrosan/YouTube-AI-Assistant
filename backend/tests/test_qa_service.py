import pytest
from langchain_core.documents import Document

from backend.services import qa_service


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.last_query = None
        self.last_k = None

    def similarity_search(self, query, k=4):
        self.last_query = query
        self.last_k = k
        return self.documents[:k]


def make_document():
    return Document(
        page_content="The video explains embeddings and vector similarity.",
        metadata={
            "video_id": "dQw4w9WgXcQ",
            "start_time": 12.0,
            "end_time": 24.0,
            "timestamp_url": ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=12s"),
        },
    )


def test_answer_question_returns_answer_and_timestamp_sources(monkeypatch):
    vectorstore = FakeVectorStore([make_document()])
    captured = {}

    def fake_generate_text(prompt, max_output_tokens):
        captured["prompt"] = prompt
        captured["max_output_tokens"] = max_output_tokens
        return "Embeddings represent data as vectors. [Source 1]"

    monkeypatch.setattr(
        qa_service,
        "generate_text",
        fake_generate_text,
    )

    result = qa_service.answer_question(
        "Explain embeddings in detail.",
        vectorstore,
    )

    assert "Embeddings represent data as vectors." in result.answer
    assert len(result.sources) == 1
    assert result.sources[0].start_time == 12.0
    assert result.sources[0].end_time == 24.0
    assert result.sources[0].timestamp_url.endswith("t=12s")
    assert vectorstore.last_k == 6
    assert "Explain embeddings in detail." in captured["prompt"]
    assert captured["max_output_tokens"] == 1800


def test_answer_question_includes_conversation_history(monkeypatch):
    vectorstore = FakeVectorStore([make_document()])
    captured = {}

    def fake_generate_text(prompt, max_output_tokens):
        captured["prompt"] = prompt
        return "Similarity search compares vectors. [Source 1]"

    monkeypatch.setattr(
        qa_service,
        "generate_text",
        fake_generate_text,
    )

    qa_service.answer_question(
        "Why is that useful?",
        vectorstore,
        chat_history=[
            {
                "role": "user",
                "content": "What are embeddings?",
            },
            {
                "role": "assistant",
                "content": "Embeddings represent data as vectors.",
            },
        ],
    )

    assert "What are embeddings?" in captured["prompt"]
    assert "Embeddings represent data as vectors." in captured["prompt"]
    assert "Why is that useful?" in captured["prompt"]


def test_answer_question_returns_fallback_when_no_documents(monkeypatch):
    vectorstore = FakeVectorStore([])

    def unexpected_call(*args, **kwargs):
        pytest.fail("Gemini should not be called without evidence.")

    monkeypatch.setattr(
        qa_service,
        "generate_text",
        unexpected_call,
    )

    result = qa_service.answer_question(
        "Explain the main idea.",
        vectorstore,
    )

    assert result.sources == ()
    assert "couldn't find relevant transcript sections" in result.answer


def test_answer_question_rejects_empty_question():
    with pytest.raises(ValueError, match="Please enter a question"):
        qa_service.answer_question(
            "   ",
            FakeVectorStore([make_document()]),
        )


def test_answer_question_rejects_invalid_k():
    with pytest.raises(ValueError, match="greater than zero"):
        qa_service.answer_question(
            "Explain the video.",
            FakeVectorStore([make_document()]),
            k=0,
        )


def test_answer_question_rejects_empty_gemini_response(monkeypatch):
    monkeypatch.setattr(
        qa_service,
        "generate_text",
        lambda prompt, max_output_tokens: "  ",
    )

    with pytest.raises(RuntimeError, match="empty answer"):
        qa_service.answer_question(
            "Explain embeddings.",
            FakeVectorStore([make_document()]),
        )
