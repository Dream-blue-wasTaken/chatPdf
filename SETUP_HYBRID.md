# Setup Guide for Hybrid RAG with Together API

This guide explains how to set up and use the hybrid RAG (Retrieval-Augmented Generation) system that combines local document processing with Together AI's DeepSeek R1 API for generating responses.

## Overview

The hybrid approach consists of:
1. **Local document processing**: PDF loading, text extraction, chunking
2. **Local embeddings**: Using HuggingFace sentence transformers for creating document embeddings
3. **Local vector storage**: ChromaDB for storing and retrieving document chunks
4. **Together API**: Using DeepSeek R1 for generating high-quality responses

## Requirements

### Python Requirements

Install the required Python packages:

```bash
pip install langchain langchain_community langchain-core langchain-huggingface together sentence-transformers chromadb pypdf streamlit streamlit-chat python-dotenv
```

The exact package list:
- `langchain` - For the RAG pipeline
- `langchain_community` - For community components like document loaders
- `langchain-core` - Core LangChain functionality
- `langchain-huggingface` - For HuggingFace embeddings integration
- `together` - Together API client
- `sentence-transformers` - For HuggingFace embedding models
- `chromadb` - Vector database for storing embeddings
- `pypdf` - For PDF processing
- `streamlit` - For the web interface
- `streamlit-chat` - For chat UI components
- `python-dotenv` - For loading environment variables from .env file

### API Keys

You need:
- **Together API key** (default one is provided in the .env file)

### Setting up the .env file

Create a file named `.env` in the root directory with your Together API key:

```
# Together API key
TOGETHER_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual Together API key.

## Implementation Files

1. **hybrid_rag.py** - Core implementation of the hybrid RAG system
2. **hybrid_sample_usage.py** - Command-line interface
3. **hybrid_app.py** - Streamlit web interface
4. **.env** - Environment variables file for API keys

## Using the System

### Command-line Interface

```bash
python hybrid_sample_usage.py --pdf your_document.pdf
```

Options:
- `--pdf`: Path to the PDF file (required)
- `--together_api_key`: Your Together API key (overrides the one in .env)
- `--together_model`: Together model name (default: deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free)
- `--embedding_model`: HuggingFace embedding model (default: all-MiniLM-L6-v2)
- `--k`: Number of chunks to retrieve (default: 5)
- `--threshold`: Similarity threshold (default: 0.2)

### Streamlit Web Interface

```bash
streamlit run hybrid_app.py
```

## Embedding Models

The system uses HuggingFace's sentence-transformers for creating embeddings. You can choose from several models:

- **all-MiniLM-L6-v2** (default): Fast and efficient with 384-dimensional embeddings, good for general text
- **all-mpnet-base-v2**: Higher quality but slower
- **paraphrase-multilingual-MiniLM-L12-v2**: Good for multilingual documents

Each embedding model gets its own ChromaDB directory to avoid dimension conflicts. These models will be automatically downloaded when first used.

## Optimizing PDF Embedding

The PDF embedding process is optimized by:

1. Using efficient HuggingFace embedding models that run locally
2. Proper chunking with 1024 tokens and 100 token overlap
3. Filtering complex metadata to reduce storage size
4. Using ChromaDB's similarity search with threshold filtering
5. Creating separate databases for each embedding model to avoid dimension conflicts

You can further optimize by:

- Adjusting chunk size based on your document's complexity
- Using a smaller embedding model for faster processing
- Setting an appropriate similarity threshold (0.2-0.3 is usually good)
- Running on a machine with a GPU for faster embedding generation

## Comparison with Other Approaches

| Approach | Embedding Generation | LLM Processing | Benefits |
|----------|---------------------|----------------|----------|
| Pure Local | Local (Ollama) | Local (Ollama) | Complete offline usage, privacy |
| Pure API | API (OpenAI) | API (OpenAI) | High quality, no local resources |
| **Hybrid** | **Local (HuggingFace)** | **API (Together)** | **Balance of efficiency and quality** |

## Troubleshooting

- If you encounter memory issues, try reducing the chunk size in hybrid_rag.py
- Make sure you have sufficient disk space for the embeddings
- The first run will download the embedding model, which may take some time
- If you get an API error, check your internet connection and API key
- If you see a dimension mismatch error, make sure you're not switching embedding models with an existing database 