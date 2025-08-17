# ChatPDF with Local DeepSeek R1

A Retrieval-Augmented Generation (RAG) application for chatting with PDF documents using locally hosted LLMs through Ollama.

## Overview

This project creates a conversational interface to PDF documents using:
- Local inference with DeepSeek R1 through Ollama
- LangChain for RAG pipeline construction
- Chroma vector database for document storage
- Streamlit for the web interface

## Features

- Upload and process PDF documents
- Chat with your documents using natural language
- Fully local operation (no API keys required)
- Configurable retrieval parameters
- Simple and intuitive UI

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) installed locally
- DeepSeek R1 model pulled in Ollama
- MxBai embedding model pulled in Ollama

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatpdf-rag-deepseek-r1.git
   cd chatpdf-rag-deepseek-r1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Pull the required models in Ollama:
   ```bash
   ollama pull deepseek-r1:latest
   ollama pull mxbai-embed-large
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Upload a PDF document through the UI

3. Ask questions about your document

4. Adjust the retrieval settings as needed:
   - Number of Retrieved Results (k): Controls how many document chunks to retrieve
   - Similarity Score Threshold: Controls the minimum relevance score for retrieved chunks

## Project Structure

- `rag.py`: Core implementation of the ChatPDF class for RAG
- `app.py`: Streamlit web application
- `chroma_db/`: Directory for storing document embeddings

## How It Works

1. **Document Ingestion**:
   - PDFs are loaded and split into smaller chunks
   - Each chunk is embedded using the MxBai embedding model
   - Embeddings are stored in a Chroma vector database

2. **Question Answering**:
   - User query is processed and relevant chunks are retrieved
   - Retrieved context and question are formatted with a prompt
   - DeepSeek R1 model generates an answer based on the context

## Core Components

### ChatPDF Class

The main class handling PDF processing and question answering:

```python
class ChatPDF:
    def __init__(self, llm_model="deepseek-r1:latest", embedding_model="mxbai-embed-large"):
        # Initialize with specified models
        
    def ingest(self, pdf_file_path):
        # Process and store PDF content
        
    def ask(self, query, k=5, score_threshold=0.2):
        # Answer questions using the RAG pipeline
        
    def clear(self):
        # Reset vector store and retriever
```

### Web Interface

The Streamlit application provides:
- Document upload interface
- Chat history display
- Configurable retrieval parameters
- User-friendly interaction

## Customization

You can customize various aspects of the application:

- **LLM Model**: Change the language model in the `ChatPDF` initialization
- **Embedding Model**: Switch to a different embedding model
- **Chunk Size**: Adjust the document chunking in the `text_splitter` initialization
- **Prompt Template**: Modify the prompt in the `ChatPDF` class

## Troubleshooting

- **Memory Issues**: If you experience memory problems, try reducing the chunk size
- **Slow Responses**: First query may be slow as the LLM loads, subsequent queries will be faster
- **Empty Responses**: Adjust the similarity threshold to a lower value
- **Irrelevant Answers**: Try increasing the `k` value to retrieve more context




