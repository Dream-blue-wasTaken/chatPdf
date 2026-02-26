#!/usr/bin/env python3
"""
Streamlit app for the API-based ChatPDF implementation.
"""
import os
import tempfile
import time
import streamlit as st
from streamlit_chat import message
from methods.api.alternative_method import APIChatPDF

st.set_page_config(page_title="RAG with OpenAI API")


def display_messages():
    """Display the chat history."""
    st.subheader("Chat History")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    """Process the user input and generate an assistant response."""
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("Thinking..."):
            try:
                agent_text = st.session_state["assistant"].ask(
                    user_text,
                    k=st.session_state["retrieval_k"],
                    score_threshold=st.session_state["retrieval_threshold"],
                )
            except ValueError as e:
                agent_text = str(e)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    """Handle file upload and ingestion."""
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}..."):
            t0 = time.time()
            st.session_state["assistant"].ingest(file_path)
            t1 = time.time()

        st.session_state["messages"].append(
            (f"Ingested {file.name} in {t1 - t0:.2f} seconds", False)
        )
        os.remove(file_path)


def page():
    """Main app page layout."""
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        # Get API key from environment or Streamlit secrets
        api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
        st.session_state["api_key"] = api_key

    st.header("RAG with OpenAI API")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key", 
        value=st.session_state.get("api_key", ""),
        type="password", 
        key="api_key_input"
    )
    
    # Model selection
    model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    selected_model = st.selectbox(
        "Select OpenAI Model",
        options=model_options,
        index=0,
        key="model_selection"
    )
    
    # Initialize assistant if API key is provided
    if api_key:
        if "assistant" not in st.session_state or st.session_state.get("current_model") != selected_model:
            st.session_state["assistant"] = APIChatPDF(
                openai_api_key=api_key,
                openai_model=selected_model
            )
            st.session_state["current_model"] = selected_model
    else:
        st.warning("Please enter your OpenAI API key to continue.")
        return

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

    # Retrieval settings
    st.subheader("Settings")
    st.session_state["retrieval_k"] = st.slider(
        "Number of Retrieved Results (k)", min_value=1, max_value=10, value=5
    )
    st.session_state["retrieval_threshold"] = st.slider(
        "Similarity Score Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05
    )

    # Display messages and text input
    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

    # Clear chat
    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.session_state["assistant"].clear()


if __name__ == "__main__":
    page() 