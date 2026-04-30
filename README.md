# TED Talks LLM (RAG)

Question answering over a TED Talks corpus using **retrieval-augmented generation (RAG)**: embeddings are stored in **Pinecone**, and answers are produced with an OpenAI-compatible API (**llmod.ai**) using only retrieved transcript context.

Repository: [TED-talks-LLM](https://github.com/omri-lazover/TED-talks-LLM)

---

## Overview

| Piece | Role |
|--------|------|
| **`embedding_pipeline_full.py`** | End-to-end offline pipeline: load CSV → chunk transcripts → embed → upsert to Pinecone (+ optional local JSON backup). |
| **`api.py`** | Flask app for **Vercel**: embeds the user question, retrieves chunks from Pinecone, calls the chat model with a strict TED-only system prompt. |
| **`chunking_text.py`**, **`embedding.py`**, **`upload_to_pinecone.py`** | Optional modular steps if you prefer not to run the full pipeline script. |

Models (see code for exact IDs) use **llmod.ai** as `base_url` with embedding and chat model names as configured in `api.py` / `embedding_pipeline_full.py`. The Pinecone index name defaults to **`ted-talks`**.

---

## Prerequisites

- Python 3.10+ (recommended)
- A **Pinecone** index (name must match `INDEX_NAME` in the code, default `ted-talks`)
- **llmod.ai** (or compatible) API key, exposed in `.env` as `OPENAI_API_KEY`
- **Pinecone** API key, `PINECONE_API_KEY`

Dataset for the pipeline: place **`ted_talks_en.csv`** in the project root (not shipped in this repo). Adjust paths in `embedding_pipeline_full.py` if your layout differs.

---

## Environment variables

Create a **`.env`** file in the project root (never commit it):

```env
OPENAI_API_KEY=your_llmod_or_compatible_key
PINECONE_API_KEY=your_pinecone_key
```

The same variables are read by the embedding pipeline and by `api.py` on Vercel (set them in the Vercel project **Environment Variables** for production).

---

## Install

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## Build the vector index (offline)

1. Ensure `ted_talks_en.csv` is present and keys are in `.env`.
2. In `embedding_pipeline_full.py`, set `PROCESS_ALL_DATA` / `ROWS_LIMIT_IF_TESTING` as needed for a full or sample run.
3. Run:

```bash
python embedding_pipeline_full.py
```

This creates a local backup JSON filename as in `BACKUP_JSON_FILE` and uploads vectors to Pinecone. If keys are missing, the script exits early with an error.

### Modular alternative

Run in order, with matching file names between steps:

1. `chunking_text.py` → chunk JSON  
2. `embedding.py` → embeddings JSON  
3. `upload_to_pinecone.py` → upsert to Pinecone  

---

## Run the API locally

```bash
python api.py
```

Default: `http://127.0.0.1:5000` (Flask `debug` as in `api.py`).

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/stats` | Returns `chunk_size`, `overlap_ratio`, `top_k` used by the service. |
| `POST` | `/api/prompt` | JSON body: `{ "question": "..." }`. Returns model answer, retrieved `context`, and `Augmented_prompt` (system + user). |

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/prompt -H "Content-Type: application/json" -d "{\"question\": \"What is the talk about education?\"}"
```

---

## Deploy (Vercel)

- Entry is configured in **`vercel.json`** (Python, `api.py`).
- Set **`OPENAI_API_KEY`** and **`PINECONE_API_KEY`** in the Vercel dashboard.
- After deploy, call the same paths on your deployment URL, e.g. `https://<project>.vercel.app/api/stats`.

---

## Project layout

```
├── api.py                      # Flask RAG API (Vercel / local)
├── vercel.json                 # Vercel build & routes
├── requirements.txt            # API + client libraries
├── embedding_pipeline_full.py  # Full index build
├── chunking_text.py            # Optional: chunking only
├── embedding.py                # Optional: embeddings only
├── upload_to_pinecone.py       # Optional: upload only
└── README.md
```
