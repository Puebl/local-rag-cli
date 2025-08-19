# Local RAG CLI

Index local documents and query them with vector search.

- Embeddings: `sentence-transformers` (default: `all-MiniLM-L6-v2`)
- Index: `hnswlib` (HNSW)
- CLI: index a folder of `.txt/.md/.pdf` (PDF optional), run queries, print latency and top-k results.

## Quickstart
```
python main.py index repo_dir path/to/docs
python main.py query repo_dir "your question" --k 5
```

Metrics: latency per query; stretch: recall@k with a labeled eval set.

Notes:
- PDF support requires `pypdf`. If not installed, PDFs are skipped.
- Runs fully locally; first run downloads the embedding model.
