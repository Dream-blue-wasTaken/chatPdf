# ChatPDF RAG Application Guide

This guide explains how the ChatPDF RAG (Retrieval-Augmented Generation) application works, covering all components, imports, and functionalities.

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [RAG Implementation (rag.py)](#rag-implementation)
5. [Streamlit Application (app.py)](#streamlit-application)
6. [How to Run](#how-to-run)
7. [Understanding RAG](#understanding-rag)

## Overview

This application allows users to upload PDF documents, ask questions about their contents, and receive accurate answers based on the document's information. It uses a RAG approach, which means:

1. The PDF document is processed, split into chunks, and stored in a vector database
2. When you ask a question, the system retrieves the most relevant document chunks
3. It then passes both your question and the relevant context to an LLM to generate an answer

The application leverages local LLM models (DeepSeek R1) through Ollama, making it possible to run entirely on your local machine without relying on external APIs.

## Project Structure

```
.
├── app.py                 # Streamlit web application
├── rag.py                 # RAG implementation using LangChain
├── requirements.txt       # Python dependencies
├── chroma_db/             # Vector database storage
└── venv/                  # Virtual environment
```

## Requirements

The application requires the following Python packages:

```python
# requirements.txt
streamlit             # Web application framework
langchain            # Framework for building LLM applications
langchain_ollama     # LangChain integration with Ollama
langchain_community  # Community components for LangChain
streamlit-chat       # Chat UI components for Streamlit
pypdf               # PDF processing library
chromadb            # Vector database for storing embeddings
```

Additionally, you need to have [Ollama](https://ollama.ai/) installed with:
- `deepseek-r1` model for text generation
- `mxbai-embed-large` model for creating embeddings

## RAG Implementation

The `rag.py` file contains the core RAG functionality through the `ChatPDF` class:

### Key Imports

```python
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
```

- `langchain_ollama`: Connects to local Ollama models
- `langchain_community.vectorstores.Chroma`: Vector database for storing document embeddings
- `PyPDFLoader`: Loads and processes PDF documents
- `RecursiveCharacterTextSplitter`: Splits documents into smaller chunks
- `ChatPromptTemplate`: Defines the prompt template for the LLM

### ChatPDF Class

The `ChatPDF` class handles:

1. **Initialization**: Setting up LLM and embedding models
   ```python
   def __init__(self, llm_model: str = "deepseek-r1:latest", embedding_model: str = "mxbai-embed-large"):
       self.model = ChatOllama(model=llm_model)
       self.embeddings = OllamaEmbeddings(model=embedding_model)
       # ...
   ```

2. **Ingestion**: Processing PDF files and storing as embeddings
   ```python
   def ingest(self, pdf_file_path: str):
       # Load PDF and split into chunks
       docs = PyPDFLoader(file_path=pdf_file_path).load()
       chunks = self.text_splitter.split_documents(docs)
       
       # Store document chunks in vector database
       self.vector_store = Chroma.from_documents(
           documents=chunks,
           embedding=self.embeddings,
           persist_directory="chroma_db",
       )
   ```
   - Documents are loaded from PDF files
   - Split into smaller chunks (1024 characters with 100 character overlap)
   - Each chunk is converted to an embedding and stored in Chroma DB

3. **Question Answering**: Retrieving context and generating answers
   ```python
   def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
       # Set up retriever if not already done
       if not self.retriever:
           self.retriever = self.vector_store.as_retriever(
               search_type="similarity_score_threshold",
               search_kwargs={"k": k, "score_threshold": score_threshold},
           )
       
       # Retrieve relevant documents
       retrieved_docs = self.retriever.invoke(query)
       
       # Format input for LLM
       formatted_input = {
           "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
           "question": query,
       }
       
       # Build and execute RAG chain
       chain = (
           RunnablePassthrough()  # Passes the input as-is
           | self.prompt           # Formats the input for the LLM
           | self.model            # Queries the LLM
           | StrOutputParser()     # Parses the LLM's output
       )
       
       return chain.invoke(formatted_input)
   ```
   - `k`: Number of most relevant chunks to retrieve (default: 5)
   - `score_threshold`: Minimum similarity score for chunks to be included (default: 0.2)
   - The retriever finds the most similar document chunks to the query
   - These chunks are joined and passed as context to the LLM
   - A LangChain runnable chain orchestrates the entire process

4. **Clearing**: Resetting the system
   ```python
   def clear(self):
       self.vector_store = None
       self.retriever = None
   ```

## Streamlit Application

The `app.py` file creates a web interface using Streamlit:

### Key Imports

```python
import os
import tempfile
import time
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF
```

- `streamlit`: Web application framework
- `streamlit_chat`: Chat UI components
- `tempfile`: For handling uploaded files
- `rag.ChatPDF`: The RAG implementation from rag.py

### Main Functions

1. **Page Layout**:
   ```python
   def page():
       st.header("RAG with Local DeepSeek R1")
       
       # File uploader
       st.subheader("Upload a Document")
       st.file_uploader(
           "Upload a PDF document",
           type=["pdf"],
           key="file_uploader",
           on_change=read_and_save_file,
           label_visibility="collapsed",
           accept_multiple_files=True,
       )
       
       # Retrieval settings
       st.subheader("Settings")
       st.session_state["retrieval_k"] = st.slider(
           "Number of Retrieved Results (k)", min_value=1, max_value=10, value=5
       )
       st.session_state["retrieval_threshold"] = st.slider(
           "Similarity Score Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05
       )
       
       # Chat display and input
       display_messages()
       st.text_input("Message", key="user_input", on_change=process_input)
       
       # Clear chat button
       if st.button("Clear Chat"):
           st.session_state["messages"] = []
           st.session_state["assistant"].clear()
   ```
   - Sets up the application layout with header, file uploader, settings sliders, and chat UI
   - Uses Streamlit session state to persist data between reruns

2. **File Processing**:
   ```python
   def read_and_save_file():
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
   ```
   - Triggers when files are uploaded
   - Creates a temporary file from the uploaded file
   - Calls the RAG ingestion process
   - Measures and displays the ingestion time
   - Removes the temporary file

3. **Chat Processing**:
   ```python
   def process_input():
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
   ```
   - Triggers when user submits a message
   - Gets answer from the RAG system with the current settings
   - Adds both user query and assistant response to the chat history

4. **Message Display**:
   ```python
   def display_messages():
       st.subheader("Chat History")
       for i, (msg, is_user) in enumerate(st.session_state["messages"]):
           message(msg, is_user=is_user, key=str(i))
       st.session_state["thinking_spinner"] = st.empty()
   ```
   - Displays all messages in the chat history
   - Uses the streamlit_chat message component for proper chat UI

## How to Run

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Install Ollama from [ollama.ai](https://ollama.ai/)

3. Pull required models:
   ```
   ollama pull deepseek-r1
   ollama pull mxbai-embed-large
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

5. Use the application:
   - Upload PDF documents
   - Ask questions about the documents
   - Adjust retrieval settings as needed

## Understanding RAG

RAG (Retrieval-Augmented Generation) combines retrieval systems with generative models. In this application:

1. **Indexing Phase**:
   - PDF is parsed and split into smaller chunks
   - Each chunk is converted to a vector embedding using mxbai-embed-large
   - Embeddings are stored in a ChromaDB vector database

2. **Retrieval Phase**:
   - User question is converted to the same embedding space
   - Vector similarity search finds the most relevant document chunks
   - Top K chunks that exceed the similarity threshold are retrieved

3. **Generation Phase**:
   - Retrieved document chunks are combined into a context
   - Context and user question are formatted into a prompt
   - DeepSeek R1 LLM generates an answer based on the prompt

The advantage of RAG is that it allows the model to "know" specific information from documents without having to fine-tune the model, reducing hallucinations and improving factual accuracy. 