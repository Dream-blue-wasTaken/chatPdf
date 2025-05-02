# API-Based ChatPDF Implementation

This is an alternative implementation of the ChatPDF application that uses OpenAI's API instead of running models locally with Ollama.

## Files

- `alternative_method.py`: The main API-based implementation class `APIChatPDF`
- `api_sample_usage.py`: A CLI script that demonstrates how to use the API implementation
- `api_app.py`: A Streamlit app similar to the original one but using the API implementation

## Requirements

To use this API-based implementation, you'll need:
https://github.com/vanzan01/cursor-memory-bank
1. An OpenAI API key
2. The following Python packages:
   - langchain
   - langchain-openai
   - langchain-community
   - langchain-core
   - openai
   - streamlit (for the web app)
   - streamlit-chat (for the web app)

You can install the requirements with:

```bash
pip install langchain langchain-openai langchain-community langchain-core openai streamlit streamlit-chat
```

## Using the API implementation

### Command Line Interface

```bash
# Basic usage
python api_sample_usage.py --pdf your_document.pdf --openai_api_key your_api_key

# Specifying model and retrieval parameters
python api_sample_usage.py --pdf your_document.pdf --model gpt-4 --k 3 --threshold 0.3
```

You can also set your API key as an environment variable:

```bash
# Unix/Linux/MacOS
export OPENAI_API_KEY=your_api_key

# Windows
set OPENAI_API_KEY=your_api_key
```

### Streamlit Web App

Run the Streamlit app with:

```bash
streamlit run api_app.py
```

In the app:
1. Enter your OpenAI API key
2. Select the model you want to use
3. Upload your PDF document
4. Adjust the retrieval settings if needed
5. Start chatting with your document

## Differences from local implementation

This API-based implementation:

1. Uses OpenAI models instead of locally hosted models via Ollama
2. Requires an API key and internet connection
3. May provide more accurate responses (depending on the model used)
4. Incurs API usage costs
5. Uses a different vector store directory (`chroma_db_api`) to avoid conflicts

## Integration

The `APIChatPDF` class provides the same interface as the original `ChatPDF` class, so you can switch between them with minimal code changes.

Example:
```python
# Local version
from rag import ChatPDF
chat = ChatPDF()

# API version
from alternative_method import APIChatPDF
chat = APIChatPDF(openai_api_key="your_api_key")

# Both versions use the same methods
chat.ingest("document.pdf")
answer = chat.ask("What is this document about?")
``` 