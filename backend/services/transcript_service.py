from dataclasses import dataclass

import requests
from youtube_transcript_api import (
    NoTranscriptFound,
    YouTubeTranscriptApi,
    YouTubeTranscriptApiException,
)

from backend.services.youtube_url_service import extract_video_id


class TranscriptServiceError(RuntimeError):
    """Raised when transcript retrieval or processing fails."""


@dataclass(frozen=True)
class TranscriptSnippet:
    text: str
    start: float
    duration: float


@dataclass(frozen=True)
class TranscriptData:
    video_id: str
    language: str
    language_code: str
    is_generated: bool
    snippets: tuple[TranscriptSnippet, ...]

    @property
    def full_text(self) -> str:
        """Return the complete transcript as readable text."""
        return " ".join(snippet.text for snippet in self.snippets)


def fetch_transcript(
    url: str,
    languages: tuple[str, ...] = ("en",),
) -> TranscriptData:
    """Retrieve available captions and preserve timestamp metadata."""

    if not languages:
        raise ValueError("At least one transcript language is required.")

    video_id = extract_video_id(url)
    api = YouTubeTranscriptApi()

    try:
        try:
            # First, try the preferred languages.
            fetched = api.fetch(
                video_id,
                languages=languages,
            )

        except NoTranscriptFound:
            # Fall back to an available language if English is unavailable.
            available = list(api.list(video_id))

            if not available:
                raise TranscriptServiceError(
                    f"No captions are available for video {video_id}."
                )

            # Prefer manually created captions, if available.
            selected = next(
                (transcript for transcript in available if not transcript.is_generated),
                available[0],
            )

            fetched = selected.fetch()

        raw_snippets = fetched.to_raw_data()

    except TranscriptServiceError:
        raise

    except (
        YouTubeTranscriptApiException,
        requests.RequestException,
    ) as exc:
        raise TranscriptServiceError(
            f"Unable to retrieve transcript for video {video_id}: {exc}"
        ) from exc

    snippets = []

    try:
        for item in raw_snippets:
            text = str(item.get("text", "")).strip()

            if not text:
                continue

            snippets.append(
                TranscriptSnippet(
                    text=text,
                    start=float(item["start"]),
                    duration=float(item["duration"]),
                )
            )

    except (KeyError, TypeError, ValueError) as exc:
        raise TranscriptServiceError(
            "The retrieved transcript contains invalid snippet data."
        ) from exc

    if not snippets:
        raise TranscriptServiceError(
            f"No transcript content was found for video {video_id}."
        )

    return TranscriptData(
        video_id=video_id,
        language=fetched.language,
        language_code=fetched.language_code,
        is_generated=fetched.is_generated,
        snippets=tuple(snippets),
    )
