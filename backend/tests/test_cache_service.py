import pytest

from backend.services import cache_service
from backend.services.transcript_service import (
    TranscriptData,
    TranscriptSnippet,
)


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cache_service,
        "VIDEO_CACHE_DIR",
        tmp_path / "videos",
    )
    monkeypatch.setattr(
        cache_service,
        "USAGE_DIR",
        tmp_path / "usage",
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
            TranscriptSnippet(
                text="The main concept is explained.",
                start=5.0,
                duration=8.0,
            ),
        ),
    )


def test_cached_video_round_trip(isolated_cache):
    transcript = make_transcript()
    overview = "# Video Overview\n\nThis video explains its main concept."

    cache_service.save_cached_video(transcript, overview)

    loaded = cache_service.load_cached_video(transcript.video_id)

    assert loaded is not None

    loaded_transcript, loaded_overview = loaded

    assert loaded_overview == overview
    assert loaded_transcript.video_id == transcript.video_id
    assert loaded_transcript.full_text == transcript.full_text
    assert loaded_transcript.snippets[1].start == 5.0
    assert loaded_transcript.language_code == "en"


def test_cache_miss_returns_none(isolated_cache):
    assert cache_service.load_cached_video("dQw4w9WgXcQ") is None


def test_daily_limit_allows_three_unique_videos(isolated_cache):
    video_ids = [
        "video000001",
        "video000002",
        "video000003",
    ]

    for video_id in video_ids:
        cache_service.reserve_daily_video_slot(video_id)
        cache_service.mark_daily_video_completed(video_id)

    with pytest.raises(cache_service.DailyVideoLimitError):
        cache_service.reserve_daily_video_slot("video000004")


def test_previously_processed_video_does_not_use_another_slot(
    isolated_cache,
):
    video_id = "video000001"

    cache_service.reserve_daily_video_slot(video_id)
    cache_service.mark_daily_video_completed(video_id)

    # Reprocessing the same ID does not consume another slot.
    cache_service.reserve_daily_video_slot(video_id)

    cache_service.reserve_daily_video_slot("video000002")
    cache_service.mark_daily_video_completed("video000002")

    cache_service.reserve_daily_video_slot("video000003")
    cache_service.mark_daily_video_completed("video000003")

    with pytest.raises(cache_service.DailyVideoLimitError):
        cache_service.reserve_daily_video_slot("video000004")


def test_failed_processing_can_release_a_slot(isolated_cache):
    cache_service.reserve_daily_video_slot("video000001")
    cache_service.release_daily_video_slot("video000001")

    # A released reservation does not consume a daily slot.
    cache_service.reserve_daily_video_slot("video000002")
    cache_service.mark_daily_video_completed("video000002")
