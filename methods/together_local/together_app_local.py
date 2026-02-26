#!/usr/bin/env python3
"""
Streamlit app for the Together API-based ChatPDF implementation with local embeddings.
No OpenAI API key required!
"""
import os
import tempfile
import time
import streamlit as st
from streamlit_chat import message
from methods.together_local.together_method_local import TogetherChatPDFLocal

st.set_page_config(page_title="RAG with Together AI (Local Embeddings)")


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
            except Exception as e:
                agent_text = f"Error: {str(e)}"

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
        together_api_key = os.environ.get("TOGETHER_API_KEY", "")
        # Try to get from secrets if environment variable is not set
        if not together_api_key:
            try:
                together_api_key = st.secrets.get("TOGETHER_API_KEY", "")
            except Exception:
                # If secrets are not configured, just use empty string
                together_api_key = ""
                
        st.session_state["together_api_key"] = together_api_key

    st.header("RAG with Together AI - DeepSeek R1 (Local Embeddings)")
    st.markdown("This version uses local embeddings from HuggingFace - no OpenAI API key needed!")
    
    # Together API Key input
    together_api_key = st.text_input(
        "Together API Key", 
        value=st.session_state.get("together_api_key", "fa80595b027f7af95287cb1ec86b2650fd7f09a63e71a596390860cafa191ed5"),
        type="password", 
        key="together_api_key_input"
    )
    
    # Model selection
    model_options = ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", "together/llama-2-7b-chat"]
    selected_model = st.selectbox(
        "Select Together AI Model",
        options=model_options,
        index=0,
        key="model_selection"
    )
    
    # Embedding model selection
    embedding_options = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"]
    selected_embedding = st.selectbox(
        "Select Embedding Model",
        options=embedding_options,
        index=0,
        key="embedding_selection"
    )
    
    # Initialize assistant if API key is provided
    if together_api_key:
        if "assistant" not in st.session_state or st.session_state.get("current_model") != selected_model or st.session_state.get("current_embedding") != selected_embedding:
            st.session_state["assistant"] = TogetherChatPDFLocal(
                together_api_key=together_api_key,
                model=selected_model,
                embedding_model=selected_embedding
            )
            st.session_state["current_model"] = selected_model
            st.session_state["current_embedding"] = selected_embedding
    else:
        st.warning("Please enter your Together API key to continue.")
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