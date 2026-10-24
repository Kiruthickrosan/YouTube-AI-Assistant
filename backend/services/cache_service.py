import json
import os
import re
import tempfile
import threading
import time
from datetime import date
from pathlib import Path

from backend.services.transcript_service import (
    TranscriptData,
    TranscriptSnippet,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
VIDEO_CACHE_DIR = DATA_DIR / "cache" / "videos"
USAGE_DIR = DATA_DIR / "cache" / "usage"

DEFAULT_DAILY_VIDEO_LIMIT = 3
RESERVATION_TTL_SECONDS = 2 * 60 * 60

_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
_LOCK = threading.RLock()


class CacheServiceError(RuntimeError):
    """Raised when the persistent cache or usage state is invalid."""


class DailyVideoLimitError(RuntimeError):
    """Raised when the daily new-video limit has been reached."""


class VideoProcessingInProgressError(RuntimeError):
    """Raised when the same video is already being processed."""


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically to reduce the risk of partial cache files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".cache-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            temporary_path = Path(handle.name)

        os.replace(temporary_path, path)

    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def _video_cache_path(video_id: str) -> Path:
    if not _VIDEO_ID_PATTERN.fullmatch(video_id or ""):
        raise ValueError("Invalid YouTube video ID.")

    return VIDEO_CACHE_DIR / f"{video_id}.json"


def load_cached_video(
    video_id: str,
) -> tuple[TranscriptData, str] | None:
    """Load a cached transcript and completed overview, if available."""

    path = _video_cache_path(video_id)

    if not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))

        if payload.get("schema_version") != 1:
            return None

        transcript_data = payload["transcript"]
        overview = payload["overview"]

        if (
            transcript_data["video_id"] != video_id
            or not isinstance(overview, str)
            or not overview.strip()
        ):
            return None

        snippets = tuple(
            TranscriptSnippet(
                text=item["text"],
                start=float(item["start"]),
                duration=float(item["duration"]),
            )
            for item in transcript_data["snippets"]
        )

        if not snippets:
            return None

        transcript = TranscriptData(
            video_id=video_id,
            language=transcript_data["language"],
            language_code=transcript_data["language_code"],
            is_generated=transcript_data["is_generated"],
            snippets=snippets,
        )

        return transcript, overview

    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
    ):
        # A damaged cache is treated as a cache miss.
        return None


def save_cached_video(
    transcript: TranscriptData,
    overview: str,
) -> None:
    """Persist transcript snippets and the completed overview."""

    if not overview or not overview.strip():
        raise ValueError("Cannot cache an empty overview.")

    payload = {
        "schema_version": 1,
        "video_id": transcript.video_id,
        "overview": overview,
        "transcript": {
            "video_id": transcript.video_id,
            "language": transcript.language,
            "language_code": transcript.language_code,
            "is_generated": transcript.is_generated,
            "snippets": [
                {
                    "text": snippet.text,
                    "start": snippet.start,
                    "duration": snippet.duration,
                }
                for snippet in transcript.snippets
            ],
        },
    }

    with _LOCK:
        _atomic_write_json(
            _video_cache_path(transcript.video_id),
            payload,
        )


def _usage_file_path() -> Path:
    # Uses the computer's local date.
    return USAGE_DIR / f"{date.today().isoformat()}.json"


def _read_usage_state(path: Path) -> dict:
    if not path.exists():
        return {
            "processed_video_ids": [],
            "in_progress": {},
        }

    try:
        state = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(state.get("processed_video_ids"), list):
            raise ValueError("Invalid processed video list.")

        if not isinstance(state.get("in_progress"), dict):
            raise ValueError("Invalid in-progress video list.")

        return state

    except (OSError, ValueError, TypeError) as exc:
        raise CacheServiceError(
            "Could not read daily usage state safely. "
            "Check the usage JSON file before continuing."
        ) from exc


def _remove_expired_reservations(state: dict, now: float) -> None:
    """Release reservations left behind by crashed processes."""

    active = {}

    for video_id, reservation in state["in_progress"].items():
        try:
            started_at = float(reservation["started_at"])
        except (KeyError, TypeError, ValueError):
            continue

        if 0 <= now - started_at < RESERVATION_TTL_SECONDS:
            active[video_id] = {"started_at": started_at}

    state["in_progress"] = active


def reserve_daily_video_slot(
    video_id: str,
    limit: int = DEFAULT_DAILY_VIDEO_LIMIT,
) -> None:
    """Reserve capacity for a new uncached video analysis."""

    if not _VIDEO_ID_PATTERN.fullmatch(video_id or ""):
        raise ValueError("Invalid YouTube video ID.")

    if limit <= 0:
        raise ValueError("Daily video limit must be greater than zero.")

    with _LOCK:
        path = _usage_file_path()
        state = _read_usage_state(path)
        now = time.time()

        _remove_expired_reservations(state, now)

        processed = list(dict.fromkeys(state["processed_video_ids"]))
        state["processed_video_ids"] = processed

        # Reprocessing a previously counted video does not use another slot.
        if video_id in processed:
            _atomic_write_json(path, state)
            return

        if video_id in state["in_progress"]:
            _atomic_write_json(path, state)
            raise VideoProcessingInProgressError(
                "This video is already being processed."
            )

        used_slots = len(set(processed) | set(state["in_progress"]))

        if used_slots >= limit:
            _atomic_write_json(path, state)
            raise DailyVideoLimitError(
                f"Daily limit reached: {limit} new videos per day. "
                "You can still open previously cached videos. "
                "Try again tomorrow."
            )

        state["in_progress"][video_id] = {"started_at": now}
        _atomic_write_json(path, state)


def mark_daily_video_completed(video_id: str) -> None:
    """Count a successfully processed video for the current day."""

    with _LOCK:
        path = _usage_file_path()
        state = _read_usage_state(path)

        state["in_progress"].pop(video_id, None)

        if video_id not in state["processed_video_ids"]:
            state["processed_video_ids"].append(video_id)

        _atomic_write_json(path, state)


def release_daily_video_slot(video_id: str) -> None:
    """Release a reservation if video processing failed."""

    with _LOCK:
        path = _usage_file_path()

        if not path.exists():
            return

        state = _read_usage_state(path)

        if video_id in state["in_progress"]:
            state["in_progress"].pop(video_id, None)
            _atomic_write_json(path, state)
