# API-Based ChatPDF Implementation

This implementation uses an OpenAI-compatible chat API with local Hugging Face embeddings.

## Files

- `alternative_method.py`: `APIChatPDF` class for ingestion + retrieval + chat.
- `api_sample_usage.py`: CLI usage for testing API RAG.
- `api_app.py`: Streamlit UI for API RAG.
- `../../hf_app.py`: root-level Hugging Face Spaces entrypoint for API mode.

## Requirements

Install project dependencies:

```bash
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run methods/api/api_app.py
```

Environment variables (optional):

- `OPENAI_API_KEY`
- `OPENAI_API_BASE` (default: `https://api.longcat.chat/openai/v1`)
- `EMBEDDING_MODEL` (default: `BAAI/bge-small-en-v1.5`)

## CLI usage

```bash
python methods/api/api_sample_usage.py --pdf your_document.pdf --openai_api_key your_api_key
```

## Hugging Face Spaces deployment (Streamlit)

1. Create a **Streamlit Space**.
2. Push this repo to the Space.
3. In Space **Settings → Variables and secrets**, add:
   - `OPENAI_API_KEY` (secret)
   - optional: `OPENAI_API_BASE`, `EMBEDDING_MODEL`
4. Set the Space app file to `hf_app.py`.
5. Keep `requirements.txt` from this repo.

This avoids requiring Ollama in Spaces and uses the API-based pipeline directly.
