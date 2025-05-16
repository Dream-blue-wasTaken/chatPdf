# ChatPDF with Together AI - DeepSeek R1 (Local Embeddings)

This project implements a RAG (Retrieval-Augmented Generation) system using Together AI's DeepSeek R1 Distill Llama 70B model, allowing you to chat with your PDF documents. It extracts content from PDFs, indexes it using local HuggingFace embeddings (no OpenAI API key required), and uses Together AI's LLM capabilities to generate contextually relevant responses to your questions.

## Features

- Upload and process PDF documents
- Extract and index content using local HuggingFace embeddings
- Retrieve relevant content using similarity search
- Generate responses using Together AI's DeepSeek R1 model
- Web interface using Streamlit
- **No OpenAI API key required!**

## Requirements

- Python 3.8+
- Together API key only (free DeepSeek R1 model available)
- Internet connection for downloading embedding models (first run only)

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

You can use the command-line interface to chat with your PDFs:

```bash
python together_sample_usage_local.py --pdf your_document.pdf
```

Options:
- `--pdf`: Path to the PDF file (required)
- `--together_api_key`: Your Together API key
- `--model`: Together model name (default: deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free)
- `--embedding_model`: HuggingFace embedding model (default: all-MiniLM-L6-v2)
- `--k`: Number of chunks to retrieve (default: 5)
- `--threshold`: Similarity threshold (default: 0.2)

### Web Interface

Run the Streamlit app with local embeddings:

```bash
streamlit run together_app_local.py
```

The app will prompt you for your Together API key if not provided in environment variables.

## Environment Variables

You can set the following environment variables:
- `TOGETHER_API_KEY`: Your Together AI API key

## Implementation Details

1. **PDF Ingestion**: The system extracts text from PDFs using PyPDFLoader, splits it into manageable chunks, and creates embeddings using local HuggingFace models.

2. **Vector Database**: Document chunks are stored in a ChromaDB vector database with their embeddings.

3. **Retrieval**: When a query is received, the system retrieves the most relevant document chunks based on similarity.

4. **Generation**: The retrieved context is sent to Together AI's DeepSeek R1 model along with the query to generate a contextually relevant response.

## Files

- `together_method_local.py`: Core implementation of the RAG system with Together AI and local embeddings
- `together_app_local.py`: Streamlit web interface
- `together_sample_usage_local.py`: Command-line sample usage

## API Key

The default Together API key included is from the example, which may have usage limitations. For production use, replace it with your own API key.

## Embedding Models

By default, the system uses the "all-MiniLM-L6-v2" model from HuggingFace for embeddings. You can change this to other models like:
- all-mpnet-base-v2 (better quality, slower)
- paraphrase-multilingual-MiniLM-L12-v2 (supports multiple languages)

The first time you run the app, it will download the embedding model from HuggingFace. This may take a few minutes, but it only happens once. 