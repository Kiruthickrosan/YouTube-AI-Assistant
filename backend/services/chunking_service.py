from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from backend.services.transcript_service import TranscriptData


def chunk_transcript(
    transcript: TranscriptData,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split a transcript into searchable documents with timestamp metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    # Build the complete transcript and record each snippet's character range.
    text_parts = []
    snippet_ranges = []
    position = 0

    for snippet in transcript.snippets:
        text = snippet.text.strip()

        if not text:
            continue

        if text_parts:
            position += 1  # Space between transcript snippets.

        snippet_start = position
        snippet_end = snippet_start + len(text)

        snippet_ranges.append(
            (
                snippet_start,
                snippet_end,
                float(snippet.start),
                float(snippet.start) + float(snippet.duration),
            )
        )

        text_parts.append(text)
        position = snippet_end

    full_text = " ".join(text_parts)

    if not full_text.strip():
        raise ValueError("Cannot create chunks from an empty transcript.")

    video_url = f"https://www.youtube.com/watch?v={transcript.video_id}"

    source_document = Document(
        page_content=full_text,
        metadata={
            "video_id": transcript.video_id,
            "source": video_url,
            "language": transcript.language,
            "language_code": transcript.language_code,
            "is_generated": transcript.is_generated,
        },
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks = splitter.split_documents([source_document])

    # Map each chunk's character range back to the original video timestamps.
    for index, chunk in enumerate(chunks):
        start_index = int(chunk.metadata.get("start_index", 0))
        end_index = start_index + len(chunk.page_content)

        overlapping_ranges = [
            item
            for item in snippet_ranges
            if item[0] < end_index and item[1] > start_index
        ]

        if not overlapping_ranges:
            raise ValueError(f"Could not map chunk {index} to transcript timestamps.")

        start_time = min(item[2] for item in overlapping_ranges)
        end_time = max(item[3] for item in overlapping_ranges)

        chunk.metadata.update(
            {
                "chunk_index": index,
                "start_time": start_time,
                "end_time": end_time,
                "timestamp_url": (f"{video_url}&t={int(start_time)}s"),
            }
        )

    return chunks
