import logging
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.qa_service import answer_question
from backend.services.vectorstore_service import build_vectorstore
from backend.services.video_service import process_video
from backend.services.cache_service import (
    DailyVideoLimitError,
    VideoProcessingInProgressError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="YouTube AI Assistant",
    page_icon="▶️",
    layout="wide",
)

st.title("YouTube AI Assistant")

# Keep the active video, vector index, and conversation in this session.
DEFAULT_STATE = {
    "video_overview": None,
    "video_id": None,
    "vectorstore": None,
    "messages": [],
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


# Analyze a video only when the form is submitted.
with st.form("video_analysis_form"):
    video_url = st.text_input(
        "YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    analyze_clicked = st.form_submit_button(
        "Analyze Video",
        type="primary",
        use_container_width=True,
    )


if analyze_clicked:
    if not video_url.strip():
        st.warning("Please enter a YouTube video URL.")
    else:
        # Clear the previous video and its conversation.
        st.session_state.video_overview = None
        st.session_state.video_id = None
        st.session_state.vectorstore = None
        st.session_state.messages = []

        try:
            with st.spinner("Analyzing the video and preparing your overview..."):
                analysis = process_video(video_url.strip())

                vectorstore = build_vectorstore(analysis.chunks)

                # Store only the overview and data required for Q&A.
                st.session_state.video_overview = analysis.overview
                st.session_state.video_id = analysis.video_id
                st.session_state.vectorstore = vectorstore

        except Exception:
            logger.exception("Video analysis failed.")

            st.error(
                "We couldn't analyze this video. Please check that its "
                "transcript is available and your Gemini configuration "
                "is correct, then try again."
            )


# Show only the finished overview as the initial video result.
if st.session_state.video_overview:
    st.markdown("## Video Overview")
    st.markdown(st.session_state.video_overview)

    st.divider()

    # Restore existing chat messages for this video.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            for source in message.get("sources", []):
                start = format_timestamp(source["start_time"])
                end = format_timestamp(source["end_time"])

                st.markdown(
                    f"[Source {source['index']} · {start}–{end}]"
                    f"({source['timestamp_url']})"
                )

    question = st.chat_input("Ask a question about this video...")

    if question:
        # Provide previous messages as context for follow-up questions.
        previous_history = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in st.session_state.messages
        ]

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Finding relevant video context..."):
                    result = answer_question(
                        question=question,
                        vectorstore=st.session_state.vectorstore,
                        chat_history=previous_history,
                        k=6,
                    )

                st.markdown(result.answer)

                sources = [
                    {
                        "index": source.index,
                        "start_time": source.start_time,
                        "end_time": source.end_time,
                        "timestamp_url": source.timestamp_url,
                    }
                    for source in result.sources
                ]

                for source in sources:
                    start = format_timestamp(source["start_time"])
                    end = format_timestamp(source["end_time"])

                    st.markdown(
                        f"[Source {source['index']} · {start}–{end}]"
                        f"({source['timestamp_url']})"
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "sources": sources,
                    }
                )

            except DailyVideoLimitError as exc:
                st.warning(str(exc))

            except VideoProcessingInProgressError as exc:
                st.info(str(exc))

            except Exception:
                logger.exception("Video analysis failed.")

            st.error(
                "We couldn't analyze this video. Please check that its "
                "transcript is available and your Gemini configuration "
                "is correct, then try again."
            )
