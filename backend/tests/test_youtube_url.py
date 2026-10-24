import pytest

from backend.services.youtube_url_service import extract_video_id

VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtube.com/watch?v={VIDEO_ID}&t=30",
        f"https://youtu.be/{VIDEO_ID}",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://youtube.com/embed/{VIDEO_ID}",
        f"https://youtube.com/live/{VIDEO_ID}",
        f"https://m.youtube.com/watch?v={VIDEO_ID}",
        f"youtube.com/watch?v={VIDEO_ID}",
    ],
)
def test_extract_video_id_from_valid_urls(url):
    assert extract_video_id(url) == VIDEO_ID


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch",
        "https://youtube.com/watch?v=invalid",
        "https://youtube.com/shorts/",
    ],
)
def test_extract_video_id_rejects_invalid_urls(url):
    with pytest.raises(ValueError):
        extract_video_id(url)


def test_extract_video_id_rejects_none():
    with pytest.raises(ValueError):
        extract_video_id(None)
