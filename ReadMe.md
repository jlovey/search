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
- `ingestion/`: Pipeline for processing raw product JSON into Qdrant.
- `pipelines/`: Advanced retrieval and cleanup utilities.
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
Process products and index them into Qdrant. This generates both dense and sparse representations.
```bash
python pipelines/ingestion/advanced_pipeline.py
```

### 2. Retrieval
Execute the query suite defined in `configs/advanced_query_config.json`.
```bash
python pipelines/retrieval/advanced_retrieval.py
```

### 3. Cleanup
Delete the collection when finished.
```bash
python pipelines/retrieval/cleanup_collection.py
```

## Configuration

- `configs/ingestion_minilm.json`: Ingestion config for MiniLM model.
- `configs/query_minilm.json`: Query config for MiniLM model.
- `configs/ingestion_e5_ml.json`: Ingestion config for Multilingual E5 model.
- `configs/query_e5_ml.json`: Query config for Multilingual E5 model.
