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

- `app.py` — Streamlit chat interface.
- `rag.py` — core ChatPDF class (ingest + ask + clear).
- `hybrid_*.py`, `api_*.py`, `together_*.py` — experimental/provider-specific variants.
- `README_*.md`, `SETUP_HYBRID.md` — variant-specific setup notes.

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

