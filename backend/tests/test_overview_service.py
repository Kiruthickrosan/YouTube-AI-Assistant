import pytest

from backend.services import overview_service
from backend.services.transcript_service import (
    TranscriptData,
    TranscriptSnippet,
)


def make_transcript(snippets=None):
    if snippets is None:
        snippets = (
            TranscriptSnippet(
                text="The video explains how vector databases work.",
                start=0.0,
                duration=6.0,
            ),
            TranscriptSnippet(
                text="It discusses embeddings and similarity search.",
                start=6.0,
                duration=8.0,
            ),
        )

    return TranscriptData(
        video_id="dQw4w9WgXcQ",
        language="English",
        language_code="en",
        is_generated=True,
        snippets=tuple(snippets),
    )


def test_generate_video_overview_returns_final_overview(monkeypatch):
    prompts = []

    def fake_generate_text(prompt, max_output_tokens):
        prompts.append(prompt)

        if "TRANSCRIPT SECTION:" in prompt:
            return "The section explains vector databases and embeddings."

        return (
            "# Video Overview\n\n"
            "The video explains vector databases, embeddings, "
            "and similarity search."
        )

    monkeypatch.setattr(
        overview_service,
        "generate_text",
        fake_generate_text,
    )

    overview = overview_service.generate_video_overview(make_transcript())

    assert overview.startswith("# Video Overview")
    assert "vector databases" in overview
    assert len(prompts) == 2
    assert "The video explains how vector databases work." in prompts[0]


def test_generate_video_overview_rejects_empty_transcript():
    transcript = make_transcript(snippets=())

    with pytest.raises(ValueError, match="empty transcript"):
        overview_service.generate_video_overview(transcript)


def test_long_transcript_is_processed_in_multiple_sections(monkeypatch):
    analyzed_sections = []

    monkeypatch.setattr(
        overview_service,
        "SECTION_MAX_CHARS",
        100,
    )

    def fake_analyze_section(
        section,
        section_number,
        total_sections,
        language,
    ):
        analyzed_sections.append(
            {
                "section": section,
                "section_number": section_number,
                "total_sections": total_sections,
                "language": language,
            }
        )
        return f"Detailed notes for section {section_number}."

    def fake_generate_text(prompt, max_output_tokens):
        return "# Video Overview\n\nCombined overview."

    monkeypatch.setattr(
        overview_service,
        "_analyze_section",
        fake_analyze_section,
    )
    monkeypatch.setattr(
        overview_service,
        "generate_text",
        fake_generate_text,
    )

    transcript = make_transcript(
        snippets=(
            TranscriptSnippet(
                text="A" * 150,
                start=0.0,
                duration=10.0,
            ),
            TranscriptSnippet(
                text="B" * 150,
                start=10.0,
                duration=10.0,
            ),
        )
    )

    overview = overview_service.generate_video_overview(transcript)

    assert len(analyzed_sections) > 1
    assert all(
        item["total_sections"] == len(analyzed_sections) for item in analyzed_sections
    )
    assert [item["section_number"] for item in analyzed_sections] == list(
        range(1, len(analyzed_sections) + 1)
    )
    assert overview == "# Video Overview\n\nCombined overview."
