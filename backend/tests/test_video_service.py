import pytest
from langchain_core.documents import Document

from backend.services import video_service
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
                text="Introduction to the video.",
                start=0.0,
                duration=5.0,
            ),
        ),
    )


def patch_cache(monkeypatch, cached=None):
    monkeypatch.setattr(
        video_service,
        "extract_video_id",
        lambda url: "dQw4w9WgXcQ",
    )
    monkeypatch.setattr(
        video_service,
        "load_cached_video",
        lambda video_id: cached,
    )
    monkeypatch.setattr(
        video_service,
        "reserve_daily_video_slot",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        video_service,
        "save_cached_video",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        video_service,
        "mark_daily_video_completed",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        video_service,
        "release_daily_video_slot",
        lambda *args, **kwargs: None,
    )


def test_process_video_returns_overview_and_chunks(monkeypatch):
    transcript = make_transcript()
    expected_chunks = [
        Document(
            page_content="Introduction to the video.",
            metadata={"video_id": transcript.video_id},
        )
    ]

    patch_cache(monkeypatch)

    monkeypatch.setattr(
        video_service,
        "fetch_transcript",
        lambda url: transcript,
    )
    monkeypatch.setattr(
        video_service,
        "chunk_transcript",
        lambda data: expected_chunks,
    )
    monkeypatch.setattr(
        video_service,
        "generate_video_overview",
        lambda data: "# Video Overview\n\nA detailed explanation.",
    )

    result = video_service.process_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    assert result.video_id == transcript.video_id
    assert result.overview.startswith("# Video Overview")
    assert result.chunks[0].page_content == expected_chunks[0].page_content
    assert not hasattr(result, "transcript")


def test_process_video_uses_cached_overview(monkeypatch):
    transcript = make_transcript()
    patch_cache(monkeypatch, cached=(transcript, "Cached overview"))

    def unexpected_call(*args, **kwargs):
        pytest.fail("Cached processing should not fetch or summarize again.")

    monkeypatch.setattr(
        video_service,
        "fetch_transcript",
        unexpected_call,
    )
    monkeypatch.setattr(
        video_service,
        "generate_video_overview",
        unexpected_call,
    )
    monkeypatch.setattr(
        video_service,
        "chunk_transcript",
        lambda data: [Document(page_content="Cached transcript chunk")],
    )

    result = video_service.process_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    assert result.overview == "Cached overview"


def test_process_video_propagates_transcript_errors(monkeypatch):
    patch_cache(monkeypatch)

    def fail_fetch(url):
        raise RuntimeError("Transcript unavailable")

    monkeypatch.setattr(
        video_service,
        "fetch_transcript",
        fail_fetch,
    )

    with pytest.raises(RuntimeError, match="Transcript unavailable"):
        video_service.process_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_process_video_rejects_empty_overview(monkeypatch):
    transcript = make_transcript()
    patch_cache(monkeypatch)

    monkeypatch.setattr(
        video_service,
        "fetch_transcript",
        lambda url: transcript,
    )
    monkeypatch.setattr(
        video_service,
        "chunk_transcript",
        lambda data: [Document(page_content="Transcript chunk")],
    )
    monkeypatch.setattr(
        video_service,
        "generate_video_overview",
        lambda data: "",
    )

    with pytest.raises(RuntimeError, match="empty result"):
        video_service.process_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
