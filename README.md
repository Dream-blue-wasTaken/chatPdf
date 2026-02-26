# ChatPDF (Local RAG + Multi-Provider Experiments)

This repository contains a **ChatPDF Retrieval-Augmented Generation (RAG)** project with multiple implementation variants:
- Local-first implementation with **Ollama + DeepSeek**
- API-based variants (Together AI / custom API)
- Hybrid retrieval examples

If you want a stable starting point, use:
- `app.py` (Streamlit UI)
- `rag.py` (core local RAG pipeline)

---

## Why this project is portfolio-ready

- End-to-end RAG workflow: ingestion, chunking, embedding, retrieval, and answer generation.
- Practical local setup for privacy-conscious document QA.
- Multiple backend variants demonstrating experimentation and adaptability.
- Clear separation between UI (`app.py`) and RAG logic (`rag.py`).

---

## Quick Start (Local Ollama Path)

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Install and pull Ollama models

```bash
ollama pull deepseek-r1:latest
ollama pull mxbai-embed-large
```

### 3) Run the app

```bash
streamlit run app.py
```

### 4) Use the UI

1. Upload one or more PDF files.
2. Ask questions grounded in the uploaded document content.
3. Tune retrieval settings:
   - **k** (number of retrieved chunks)
   - **score threshold** (minimum similarity relevance)

---

## Project Structure

- `app.py` — main Streamlit chat interface (local Ollama path).
- `rag.py` — core ChatPDF class for the main local method.
- `methods/api/` — OpenAI API-based method (`api_app.py`, `api_sample_usage.py`, `alternative_method.py`).
- `methods/together/` — Together AI method (`together_app.py`, `together_sample_usage.py`, `together_method.py`).
- `methods/together_local/` — Together AI + local embeddings method.
- `methods/hybrid/` — Hybrid retrieval method and setup guides.

This keeps the **main method in the repository root** while organizing each alternative method in its own folder.

---

## Can I host this on Hugging Face Spaces?

Yes — this project can be hosted on **Hugging Face Spaces**.

### Recommended deployment target

- Deploy `app.py` as a **Streamlit Space** if you want the main local-first UX.
- If your Space cannot run local Ollama, deploy an API-based variant instead (for example `methods/together/together_app.py` or `methods/api/api_app.py`).

### Basic steps

1. Create a new Space on Hugging Face and choose **Streamlit**.
2. Push this repository to the Space (or upload files).
3. In Space settings, add required secrets (e.g., `TOGETHER_API_KEY`, `OPENAI_API_KEY`).
4. Set the app entrypoint in your Space to one of:
   - `app.py` (main local method)
   - `methods/together/together_app.py`
   - `methods/api/api_app.py`
5. Ensure `requirements.txt` includes all dependencies used by the selected method.

If you want, I can also add a ready-to-use **Hugging Face Space configuration** file next.

---

## Cleanup Notes

To keep the repository clean and recruiter-friendly:
- Ignore generated vector databases (Chroma artifacts).
- Avoid committing local runtime folders (`venv`, caches, `.env`).
- Keep one primary "happy path" (`app.py` + `rag.py`) and present others as experiments.

---

## Suggested Resume Bullets

- Built a local-first ChatPDF application using LangChain, Chroma, Ollama, and Streamlit.
- Implemented a configurable RAG pipeline with PDF ingestion, chunked embeddings, and threshold-based retrieval.
- Developed multiple inference backends (local and API-based) to compare quality/cost/latency tradeoffs.
