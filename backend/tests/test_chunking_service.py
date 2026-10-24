import pytest

from backend.services.chunking_service import chunk_transcript
from backend.services.transcript_service import (
    TranscriptData,
    TranscriptSnippet,
)


def make_transcript():
    return TranscriptData(
        video_id="dQw4w9WgXcQ",
        language="English",
        language_code="en",
        is_generated=True,
        snippets=(
            TranscriptSnippet(
                text="Introduction to the core idea. " * 8,
                start=0.0,
                duration=20.0,
            ),
            TranscriptSnippet(
                text="The next section explains the main example. " * 8,
                start=20.0,
                duration=30.0,
            ),
        ),
    )


def test_chunk_transcript_preserves_video_metadata():
    chunks = chunk_transcript(
        make_transcript(),
        chunk_size=100,
        chunk_overlap=15,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.page_content.strip()
        assert chunk.metadata["video_id"] == "dQw4w9WgXcQ"
        assert chunk.metadata["source"].startswith("https://")
        assert chunk.metadata["chunk_index"] >= 0
        assert chunk.metadata["start_time"] >= 0
        assert chunk.metadata["end_time"] >= chunk.metadata["start_time"]
        assert "timestamp_url" in chunk.metadata


def test_chunk_transcript_preserves_timestamps():
    chunks = chunk_transcript(
        make_transcript(),
        chunk_size=100,
        chunk_overlap=15,
    )

    start_times = {chunk.metadata["start_time"] for chunk in chunks}

    assert 0.0 in start_times
    assert 20.0 in start_times


def test_chunk_transcript_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_transcript(
            make_transcript(),
            chunk_size=100,
            chunk_overlap=100,
        )


def test_chunk_transcript_rejects_empty_transcript():
    empty_transcript = TranscriptData(
        video_id="dQw4w9WgXcQ",
        language="English",
        language_code="en",
        is_generated=False,
        snippets=(),
    )

    with pytest.raises(ValueError, match="empty transcript"):
        chunk_transcript(empty_transcript)
