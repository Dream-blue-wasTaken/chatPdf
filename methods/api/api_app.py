#!/usr/bin/env python3
"""
Streamlit app for the API-based ChatPDF implementation.
Credentials are loaded from environment variables (e.g. .env.local).
"""

import os
import sys
import tempfile
import time

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

# Load .env.local (or .env) from the project root
load_dotenv(
    dotenv_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env.local",
    )
)

# Add the project root to the path so that 'methods' can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from methods.api.alternative_method import APIChatPDF

st.set_page_config(page_title="RAG with LongCat API")


def display_messages() -> None:
    """Display the chat history."""
    st.subheader("Chat History")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input() -> None:
    """Process the user input and generate an assistant response."""
    user_input = st.session_state.get("user_input", "")
    if not user_input or len(user_input.strip()) == 0:
        return

    user_text = user_input.strip()

    # Clear the input box immediately
    st.session_state["user_input"] = ""

    with st.session_state["thinking_spinner"], st.spinner("Thinking..."):
        try:
            agent_text = st.session_state["assistant"].ask(
                user_text,
                k=st.session_state["retrieval_k"],
                score_threshold=st.session_state["retrieval_threshold"],
            )
        except ValueError as error:
            agent_text = str(error)

    st.session_state["messages"].append((user_text, True))
    st.session_state["messages"].append((agent_text, False))


def read_and_save_file() -> None:
    """Handle file upload and ingestion.

    Streamlit fires on_change with ALL currently selected files whenever the
    uploader changes (add or remove). We track which filenames were already
    ingested so we only process genuinely new files.

    If a file was *removed* from the uploader we must rebuild from scratch
    because Chroma doesn't support selective document deletion by source.
    """
    current_files = {f.name: f for f in st.session_state["file_uploader"]}
    current_names = set(current_files.keys())
    already_ingested = st.session_state.get("ingested_files", set())

    # Detect removals — if any previously ingested file is no longer selected,
    # clear everything and re-ingest all remaining files from scratch.
    removed = already_ingested - current_names
    if removed:
        st.session_state["assistant"].clear()
        st.session_state["ingested_files"] = set()
        st.session_state["messages"] = []
        st.session_state["user_input"] = ""
        files_to_ingest = list(current_files.values())
    else:
        # Only ingest files that haven't been seen before
        files_to_ingest = [
            f for name, f in current_files.items() if name not in already_ingested
        ]

    for file in files_to_ingest:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.getbuffer())
            file_path = temp_file.name

        try:
            with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}..."):
                start_time = time.time()
                st.session_state["assistant"].ingest(file_path)
                end_time = time.time()

            st.session_state["ingested_files"].add(file.name)
            st.session_state["messages"].append(
                (f"Ingested {file.name} in {end_time - start_time:.2f} seconds", False)
            )
        except Exception as exc:
            st.session_state["messages"].append((f"Error ingesting {file.name}: {exc}", False))
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)


def _init_assistant() -> None:
    """Initialize APIChatPDF using env-loaded credentials."""
    if "assistant" in st.session_state:
        return

    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.longcat.chat/openai/v1")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    chat_model = os.environ.get("OPENAI_MODEL", "LongCat-Flash-Lite")

    if not api_key:
        st.error(
            "No API key found. Set `OPENAI_API_KEY` in your `.env.local` file and restart."
        )
        st.stop()

    st.session_state["assistant"] = APIChatPDF(
        openai_api_key=api_key,
        openai_model=chat_model,
        openai_api_base=api_base,
        embedding_model=embedding_model,
    )


def page() -> None:
    """Main app page layout."""
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["retrieval_k"] = 5
        st.session_state["retrieval_threshold"] = 0.2
        st.session_state["ingested_files"] = set()

    st.header("RAG with OpenAI-Compatible API")

    _init_assistant()

    st.subheader("Upload a Document")
    st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    st.subheader("Settings")
    st.session_state["retrieval_k"] = st.slider(
        "Number of Retrieved Results (k)", min_value=1, max_value=10, value=st.session_state["retrieval_k"]
    )
    st.session_state["retrieval_threshold"] = st.slider(
        "Similarity Score Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["retrieval_threshold"],
        step=0.05,
    )

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.session_state["assistant"].clear()


if __name__ == "__main__":
    page()
