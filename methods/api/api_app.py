#!/usr/bin/env python3
"""
Streamlit app for the API-based ChatPDF implementation.
"""

import os
import sys
import tempfile
import time

import streamlit as st
from streamlit_chat import message

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
    """Handle file upload and ingestion."""
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.getbuffer())
            file_path = temp_file.name

        try:
            with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}..."):
                start_time = time.time()
                st.session_state["assistant"].ingest(file_path)
                end_time = time.time()

            st.session_state["messages"].append(
                (f"Ingested {file.name} in {end_time - start_time:.2f} seconds", False)
            )
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)


def _init_assistant(api_key: str, api_base: str, selected_model: str, embedding_model: str) -> None:
    """Initialize APIChatPDF and keep state in sync when settings change."""
    needs_reinit = (
        "assistant" not in st.session_state
        or st.session_state.get("current_model") != selected_model
        or st.session_state.get("current_api_base") != api_base
        or st.session_state.get("current_api_key") != api_key
        or st.session_state.get("current_embedding_model") != embedding_model
    )

    if not needs_reinit:
        return

    st.session_state["assistant"] = APIChatPDF(
        openai_api_key=api_key,
        openai_model=selected_model,
        openai_api_base=api_base,
        embedding_model=embedding_model,
    )
    st.session_state["current_model"] = selected_model
    st.session_state["current_api_base"] = api_base
    st.session_state["current_api_key"] = api_key
    st.session_state["current_embedding_model"] = embedding_model


def page() -> None:
    """Main app page layout."""
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["api_key"] = os.environ.get("OPENAI_API_KEY", "")
        st.session_state["api_base"] = os.environ.get(
            "OPENAI_API_BASE", "https://api.longcat.chat/openai/v1"
        )
        st.session_state["embedding_model"] = os.environ.get(
            "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        )

    st.header("RAG with OpenAI-Compatible API")

    api_key = st.text_input(
        "API Key",
        value=st.session_state.get("api_key", ""),
        type="password",
        key="api_key_input",
    ).strip()

    api_base = st.text_input(
        "API Base URL",
        value=st.session_state.get("api_base", "https://api.longcat.chat/openai/v1"),
        key="api_base_input",
    ).strip()

    embedding_model = st.text_input(
        "Embedding Model (Hugging Face)",
        value=st.session_state.get("embedding_model", "BAAI/bge-small-en-v1.5"),
        key="embedding_model_input",
    ).strip()

    model_options = ["LongCat-Flash-Lite", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    selected_model = st.selectbox(
        "Select Model",
        options=model_options,
        index=0,
        key="model_selection",
    )

    if not api_key:
        st.warning("Please enter your API key to continue.")
        return

    _init_assistant(api_key, api_base, selected_model, embedding_model)

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
        "Number of Retrieved Results (k)", min_value=1, max_value=10, value=5
    )
    st.session_state["retrieval_threshold"] = st.slider(
        "Similarity Score Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05
    )

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.session_state["assistant"].clear()


if __name__ == "__main__":
    page()
