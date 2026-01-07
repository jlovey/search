# Advanced Product Search with Qdrant

This project implements a professional, full-fledged search application using Qdrant. It supports hybrid search (Dense + Sparse) with automated vectorization and RRF (Reciprocal Rank Fusion).

## Features
- **Hybrid Search**: Combines semantic understanding (Dense) with keyword matching (Sparse).
- **Automated Embedding**: Uses `sentence-transformers` for dense vectors and `TF-IDF` for sparse vectors.
- **Config-Driven Ingestion**: Define which fields to embed and filter via JSON.
- **Flexible Retrieval**: Supports pure dense, pure sparse, and hybrid search modes.
- **RRF Fusion**: Intelligently merges results from different retrieval methods.

## Project Structure
- `src/services/`: Core logic for `Qdrant` and metrics.
- `models/`: Embedding logic and persistent model storage.
- `pipelines/ingestion/models/`: Model-specific indexing logic (MiniLM, E5, etc.).
- `pipelines/retrieval/models/`: Model-specific search and reranking logic.
- `pipelines/ingestion/ingestion_orchestrator.py`: Entry point for indexing.
- `pipelines/retrieval/retrieval_orchestrator.py`: Entry point for search.
- `configs/`: Centralized configuration for ingestion and queries.
- `data/`: Sample product datasets.

## Setup

1. **Start Qdrant**:
   ```bash
   docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

2. **Initialize Environment**:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### 1. Ingestion
Process products and index them into Qdrant using the model-specific ingestors.
```bash
python pipelines/ingestion/ingestion_orchestrator.py
```

### 2. Retrieval
Execute queries using the model-specific retrievers.
```bash
python pipelines/retrieval/retrieval_orchestrator.py
```

### 3. Cleanup
Delete the collection when finished.
```bash
python pipelines/retrieval/cleanup_collection.py
```

### 4. Testing
Run the automated test suite across all configured models.
```bash
pytest -v -s tests/scripts/test_search_pipeline.py
```

## Configuration
- `configs/ingestion_minilm.json`: Ingestion config for MiniLM model.
- `configs/query_minilm.json`: Query config for MiniLM model.
- `configs/ingestion_e5_ml.json`: Ingestion config for Multilingual E5 model.
- `configs/query_e5_ml.json`: Query config for Multilingual E5 model.
- `tests/configs/test_runtime_config.json`: Master configuration for automated tests.
