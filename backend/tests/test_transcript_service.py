import pytest

from backend.services import transcript_service as service


class FakeFetchedTranscript:
    language = "English"
    language_code = "en"
    is_generated = True

    def to_raw_data(self):
        return [
            {"text": "Introduction", "start": 0.0, "duration": 2.5},
            {"text": "Main concept", "start": 2.5, "duration": 4.0},
        ]


class FakeTranscriptAPI:
    def fetch(self, video_id, languages=("en",)):
        assert video_id == "dQw4w9WgXcQ"
        assert languages == ("en",)
        return FakeFetchedTranscript()


def test_fetch_transcript_preserves_metadata(monkeypatch):
    monkeypatch.setattr(
        service,
        "YouTubeTranscriptApi",
        lambda: FakeTranscriptAPI(),
    )

    result = service.fetch_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    assert result.video_id == "dQw4w9WgXcQ"
    assert result.language_code == "en"
    assert result.is_generated is True
    assert result.full_text == "Introduction Main concept"
    assert result.snippets[1].start == 2.5
    assert result.snippets[1].duration == 4.0


def test_fetch_transcript_rejects_invalid_url():
    with pytest.raises(ValueError):
        service.fetch_transcript("https://example.com/video")


def test_fetch_transcript_rejects_empty_languages():
    with pytest.raises(ValueError):
        service.fetch_transcript(
            "https://youtu.be/dQw4w9WgXcQ",
            languages=(),
        )


def test_fetch_transcript_rejects_empty_content(monkeypatch):
    class EmptyTranscript(FakeFetchedTranscript):
        def to_raw_data(self):
            return []

    class EmptyAPI:
        def fetch(self, video_id, languages=("en",)):
            return EmptyTranscript()

    monkeypatch.setattr(
        service,
        "YouTubeTranscriptApi",
        lambda: EmptyAPI(),
    )

    with pytest.raises(service.TranscriptServiceError):
        service.fetch_transcript("https://youtu.be/dQw4w9WgXcQ")
