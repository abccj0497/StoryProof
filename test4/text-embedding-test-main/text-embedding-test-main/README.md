# Text Embedding Test

A web application for comparing text embedding models and measuring similarity.

## Features

- Generate embedding vectors for multiple sentences
- Compare various embedding models
- Calculate cosine similarity and Euclidean distance
- Real-time similarity matrix visualization

## Supported Models

- Qwen/Qwen3-Embedding-0.6B (1024 dimensions)
- sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)

**Note**: Any SentenceTransformer-compatible embedding model can be dynamically added by modifying the `EMBEDDING_MODELS` list in `app/routers/embedding.py`.

```python
EMBEDDING_MODELS = [
    {
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "dimension": 1024,
    },
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
    }
]
```

## Requirements

- Python 3.12 (recommended)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Server

```bash
uvicorn app.main:app --reload
```

Or specify host and port:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access via Web Browser

Once the server is running, visit:

```
http://localhost:8000
```