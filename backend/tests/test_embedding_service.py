import pytest

from backend.services import embedding_service


def test_get_embeddings_uses_config_and_caches(monkeypatch):
    calls = []

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(kwargs)

    monkeypatch.setenv("EMBEDDING_MODEL", "test/embedding-model")
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")

    monkeypatch.setattr(
        embedding_service,
        "HuggingFaceEmbeddings",
        FakeEmbeddings,
    )

    embedding_service.get_embeddings.cache_clear()

    try:
        first = embedding_service.get_embeddings()
        second = embedding_service.get_embeddings()

        assert first is second
        assert len(calls) == 1
        assert calls[0]["model_name"] == "test/embedding-model"
        assert calls[0]["model_kwargs"] == {"device": "cpu"}
        assert calls[0]["encode_kwargs"]["normalize_embeddings"] is True
    finally:
        embedding_service.get_embeddings.cache_clear()


def test_get_embeddings_rejects_empty_model(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", " ")

    embedding_service.get_embeddings.cache_clear()

    try:
        with pytest.raises(RuntimeError, match="cannot be empty"):
            embedding_service.get_embeddings()
    finally:
        embedding_service.get_embeddings.cache_clear()
