from dataclasses import dataclass

from langchain_core.documents import Document

from backend.services.cache_service import (
    DEFAULT_DAILY_VIDEO_LIMIT,
    load_cached_video,
    mark_daily_video_completed,
    release_daily_video_slot,
    reserve_daily_video_slot,
    save_cached_video,
)
from backend.services.chunking_service import chunk_transcript
from backend.services.overview_service import generate_video_overview
from backend.services.transcript_service import fetch_transcript
from backend.services.youtube_url_service import extract_video_id


@dataclass(frozen=True)
class VideoAnalysisResult:
    """Video information needed by the frontend and Q&A pipeline."""

    video_id: str
    overview: str
    language: str
    language_code: str
    chunks: tuple[Document, ...]


def process_video(url: str) -> VideoAnalysisResult:
    """Analyze a new video or reuse its cached transcript and overview."""

    video_id = extract_video_id(url)

    # Check cache first: cached videos do not consume a daily slot
    # and do not require another transcript fetch or overview generation.
    cached = load_cached_video(video_id)

    if cached is not None:
        transcript, overview = cached
        chunks = tuple(chunk_transcript(transcript))

        if not chunks:
            raise RuntimeError("The cached transcript produced no chunks.")

        return VideoAnalysisResult(
            video_id=video_id,
            overview=overview,
            language=transcript.language,
            language_code=transcript.language_code,
            chunks=chunks,
        )

    # Reserve capacity before any external processing begins.
    reserve_daily_video_slot(
        video_id,
        limit=DEFAULT_DAILY_VIDEO_LIMIT,
    )

    try:
        transcript = fetch_transcript(url)

        if transcript.video_id != video_id:
            raise RuntimeError(
                "Transcript video ID does not match the requested video."
            )

        # Prepare chunks before spending Gemini requests.
        chunks = tuple(chunk_transcript(transcript))

        if not chunks:
            raise RuntimeError("The transcript produced no searchable chunks.")

        overview = generate_video_overview(transcript)

        if not overview or not overview.strip():
            raise RuntimeError("Overview generation returned an empty result.")

        # Cache only after the transcript, chunks and overview are ready.
        save_cached_video(transcript, overview)
        mark_daily_video_completed(video_id)

        return VideoAnalysisResult(
            video_id=video_id,
            overview=overview,
            language=transcript.language,
            language_code=transcript.language_code,
            chunks=chunks,
        )

    except Exception:
        release_daily_video_slot(video_id)
        raise
