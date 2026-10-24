import os
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load and cache the embedding model for reuse."""

    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    ).strip()

    device = os.getenv("EMBEDDING_DEVICE", "cpu").strip()

    if not model_name:
        raise RuntimeError("EMBEDDING_MODEL cannot be empty.")

    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("EMBEDDING_DEVICE must be cpu, cuda, or mps.")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
