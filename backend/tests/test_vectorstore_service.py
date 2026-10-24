import pytest
from langchain_core.documents import Document

from backend.services import vectorstore_service


def test_build_vectorstore_uses_embeddings_and_preserves_chunks(monkeypatch):
    chunks = [
        Document(
            page_content="Vector databases store embeddings.",
            metadata={
                "video_id": "video123456",
                "start_time": 10.0,
                "end_time": 18.0,
            },
        )
    ]

    fake_embeddings = object()
    expected_store = object()
    captured = {}

    def fake_from_documents(documents, embedding):
        captured["documents"] = documents
        captured["embedding"] = embedding
        return expected_store

    monkeypatch.setattr(
        vectorstore_service,
        "get_embeddings",
        lambda: fake_embeddings,
    )
    monkeypatch.setattr(
        vectorstore_service.FAISS,
        "from_documents",
        staticmethod(fake_from_documents),
    )

    result = vectorstore_service.build_vectorstore(chunks)

    assert result is expected_store
    assert captured["documents"] == chunks
    assert captured["embedding"] is fake_embeddings
    assert captured["documents"][0].metadata["start_time"] == 10.0


def test_build_vectorstore_rejects_empty_chunks():
    with pytest.raises(ValueError, match="without chunks"):
        vectorstore_service.build_vectorstore([])


def test_build_vectorstore_rejects_blank_documents():
    chunks = [Document(page_content="  ")]

    with pytest.raises(ValueError, match="All transcript chunks are empty"):
        vectorstore_service.build_vectorstore(chunks)


def test_create_retriever_uses_requested_k():
    class FakeVectorStore:
        def as_retriever(self, **kwargs):
            return kwargs

    result = vectorstore_service.create_retriever(
        FakeVectorStore(),
        k=5,
    )

    assert result["search_type"] == "similarity"
    assert result["search_kwargs"] == {"k": 5}


def test_create_retriever_rejects_invalid_k():
    with pytest.raises(ValueError, match="greater than zero"):
        vectorstore_service.create_retriever(
            object(),
            k=0,
        )
