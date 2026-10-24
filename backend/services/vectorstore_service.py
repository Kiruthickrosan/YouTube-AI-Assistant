from collections.abc import Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.services.embedding_service import get_embeddings


def build_vectorstore(chunks: Sequence[Document]) -> FAISS:
    """Create a FAISS index from transcript chunks."""

    if not chunks:
        raise ValueError("Cannot build a vector store without chunks.")

    valid_chunks = []

    for chunk in chunks:
        if not isinstance(chunk, Document):
            raise TypeError("Every chunk must be a LangChain Document.")

        if chunk.page_content.strip():
            valid_chunks.append(chunk)

    if not valid_chunks:
        raise ValueError("All transcript chunks are empty.")

    return FAISS.from_documents(
        documents=valid_chunks,
        embedding=get_embeddings(),
    )


def create_retriever(vectorstore: FAISS, k: int = 6):
    """Return a retriever configured to find relevant transcript chunks."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
