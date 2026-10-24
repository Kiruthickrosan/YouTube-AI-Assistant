import re
from urllib.parse import parse_qs, urlparse

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
    "youtu.be",
    "www.youtu.be",
}

SHORT_URL_HOSTS = {"youtu.be", "www.youtu.be"}

VIDEO_PATH_PREFIXES = {"embed", "shorts", "live", "v"}

VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url: str) -> str:
    """Validate a YouTube URL and extract its 11-character video ID."""

    if not isinstance(url, str) or not url.strip():
        raise ValueError("Please enter a YouTube video URL.")

    url = url.strip()

    # Allow URLs pasted without an explicit scheme.
    if "://" not in url:
        url = f"https://{url}"

    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
    except ValueError as exc:
        raise ValueError("Invalid YouTube URL.") from exc

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("The URL must use HTTP or HTTPS.")

    if hostname not in YOUTUBE_HOSTS:
        raise ValueError("Please provide a URL from YouTube.")

    video_id = None
    path = parsed.path.rstrip("/")

    if hostname in SHORT_URL_HOSTS:
        parts = path.strip("/").split("/")
        video_id = parts[0] if parts and parts[0] else None

    elif path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [None])[0]

    else:
        parts = path.strip("/").split("/")

        if len(parts) >= 2 and parts[0] in VIDEO_PATH_PREFIXES:
            video_id = parts[1]

    if not video_id or not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("Could not extract a valid YouTube video ID from this URL.")

    return video_id
