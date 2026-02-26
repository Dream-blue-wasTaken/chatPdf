# ChatPDF with Together AI - DeepSeek R1

This project implements a RAG (Retrieval-Augmented Generation) system using Together AI's DeepSeek R1 Distill Llama 70B model, allowing you to chat with your PDF documents. It extracts content from PDFs, indexes it, and uses Together AI's LLM capabilities to generate contextually relevant responses to your questions.

## Features

- Upload and process PDF documents
- Extract and index content using OpenAI embeddings
- Retrieve relevant content using similarity search
- Generate responses using Together AI's DeepSeek R1 model
- Web interface using Streamlit

## Requirements

- Python 3.8+
- Together API key
- OpenAI API key (for embeddings)

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
python together_sample_usage.py --pdf your_document.pdf
```

Options:
- `--pdf`: Path to the PDF file (required)
- `--together_api_key`: Your Together API key
- `--openai_api_key`: Your OpenAI API key (for embeddings)
- `--model`: Together model name (default: deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free)
- `--k`: Number of chunks to retrieve (default: 5)
- `--threshold`: Similarity threshold (default: 0.2)

### Web Interface

Run the Streamlit app:

```bash
streamlit run together_app.py
```

The app will prompt you for your API keys if not provided in environment variables.

### Simple Example

A simple example script (`together_simple_example.py`) demonstrates direct usage of the Together API:

```bash
python together_simple_example.py
```

## Environment Variables

You can set the following environment variables:
- `TOGETHER_API_KEY`: Your Together AI API key
- `OPENAI_API_KEY`: Your OpenAI API key (for embeddings)

## Implementation Details

1. **PDF Ingestion**: The system extracts text from PDFs using PyPDFLoader, splits it into manageable chunks, and creates embeddings.

2. **Vector Database**: Document chunks are stored in a ChromaDB vector database with their embeddings.

3. **Retrieval**: When a query is received, the system retrieves the most relevant document chunks based on similarity.

4. **Generation**: The retrieved context is sent to Together AI's DeepSeek R1 model along with the query to generate a contextually relevant response.

## Files

- `together_method.py`: Core implementation of the RAG system with Together AI
- `together_app.py`: Streamlit web interface
- `together_sample_usage.py`: Command-line sample usage
- `together_simple_example.py`: Simple example of direct Together API usage

## API Key

The default Together API key included is from the example, which may have usage limitations. For production use, replace it with your own API key. 