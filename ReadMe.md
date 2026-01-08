# Advanced Product Search with Qdrant

This project implements a professional, full-fledged search application using Qdrant. It supports hybrid search (Dense + Sparse) with automated vectorization and RRF (Reciprocal Rank Fusion).

## Features
- **Hybrid Search**: Combines semantic understanding (Dense) with keyword matching (Sparse).
- **Automated Embedding**: Uses `sentence-transformers` for dense vectors and `TF-IDF` for sparse vectors.
- **Config-Driven Ingestion**: Define which fields to embed and filter via JSON.
- **Flexible Retrieval**: Supports pure dense, pure sparse, and hybrid search modes.
- **RRF Fusion**: Intelligently merges results from different retrieval methods.
- **Score Normalization**: All search results (Dense, Sparse, Hybrid) are normalized to a consistent `[0, 1]` range for easy comparison.
- **Enrichment Engine**: Modular preparation for dense (Markdown) and sparse (flattened) vectors.
- **Folder Ingestion**: Supports both single JSON files and directories containing multiple JSON files, with automated cross-file sparse model fitting.
- **UTF-8 Compliance**: Guaranteed thread-safe, cross-platform string handling by enforcing UTF-8 encoding across all file I/O operations.

## Scoring & Normalization

The pipeline implements a dedicated normalization layer in the `BaseRetriever` to make scores intuitive and comparable:

1.  **Dense & Sparse**: Uses **Min-Max scaling** on the raw similarity scores within each result set. This maps the results into a `[0, 1]` range relative to the top performer of that specific query.
2.  **Hybrid (RRF)**: Reciprocal Rank Fusion scores are normalized by dividing the total rank-score by the theoretical maximum for a 2-stream search (`~0.0327`). 
3.  **Transparency**: The `retrieval_orchestrator` outputs both the **Normalized Score** (for business logic/thresholds) and the **Raw Score** (for debugging and technical analysis).

## Enrichment Pipeline

The system uses a dedicated `EnrichmentService` to prepare data for vectorization, ensuring the best possible retrieval performance:

- **Dense Enrichment (Markdown formatting)**:
  - Higher semantic quality for LLM-based dense models (MiniLM, E5).
  - Automatically formats products into structured Markdown:
    ```markdown
    # [ProductName]
    [ProductDescription]
    ---
    [Attribute]: [Value]
    ```
  - All content is lowercased to maintain embedding consistency and reduce noise.
- **Sparse Enrichment (Flattening)**:
  - Optimized for keyword-based search (BM25/TF-IDF).
  - Flattens specified fields into a continuous, lowercase text stream for efficient tokenization and vocabulary mapping.

## Project Structure

```text
search/
├── configs/                # JSON configurations for each model
│   ├── ingestion_*.json    # Model-specific ingestion settings
│   └── query_*.json        # Model-specific query scenarios
├── data/                   # Dataset storage (Sample and Production)
├── models/                 # Shared embedding and sparse model services
├── pipelines/              
│   ├── ingestion/          # Orchestrators and model-specific ingestors
│   └── retrieval/          # Search logic and model-specific retrievers
├── src/services/           # Core helpers (Qdrant, Enrichment, Metrics)
├── tests/
│   ├── configs/            # Runtime test configuration
│   ├── results/            # Captured logs and comparison reports
│   └── scripts/            # Comprehensive test suite
│       ├── [model_name]/   # Full cycle, Ingest-only, Query-only per model
│       ├── bulk_ingest_all.py # Sequential ingestion for all models
│       └── bulk_query_all.py  # Sequential query testing for all models
├── requirements.txt        # Project dependencies
└── ReadMe.md               # You are here
```

### **Core Component Descriptions**

*   **`configs/`**: The "brain" of the project where you manage vector dimensions, model-specific prefixes (e.g., `passage:` for E5), and search thresholds.
*   **`pipelines/`**: Implements the orchestration pattern. It dynamically routes requests to model-specific logic, ensuring internal changes to one model don't affect others.
*   **`src/services/`**: Infrastructure layer handling Qdrant interactions and search performance metrics.
*   **`tests/`**: Robust suite of 5 test scenarios for verifying pipeline integrity across different model deployments.

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

## Usage & Bulk Operations

The system is designed for high-throughput evaluation. You can process single models or the entire suite.

### 1. Bulk Benchmarking (Production)
Run the entire pipeline across all 6 models with performance tracking:
```bash
# Sequential ingestion for all models
python tests/scripts/bulk_ingest_all.py

# Sequential query testing for all models (with Avg Query Latency)
python tests/scripts/bulk_query_all.py
```

### 2. Standardized Testing
Each model has a dedicated test suite under `tests/scripts/[model_name]/`:
- `test_full_cycle.py`: Purge → Index → Search → Cleanup.
- `test_ingest_only.py`: Purge → Index.
- `test_query_only.py`: Search existing collection.

To run model-specific tests:
```bash
pytest -v -s tests/scripts/qwen3/test_full_cycle.py
```

## Performance & Comparison Analysis

The project provides automated tools to analyze model performance:

1.  **Time Tracking**: Every test script captures **Suite Time** and **Average Latency** (per-query).
2.  **Rank Comparison**: Running `bulk_query_all.py` allows for generating side-by-side position analysis reports in `tests/results/model_comparison.md`.

## Supported Models

1.  **Qwen3-Embedding-0.6B**: High-performance instruction-based model.
2.  **BAAI/bge-small-en-v1.5**: Best-in-class accuracy for size.
3.  **mixedbread-ai/mxbai-embed-large-v1**: Premium dense retrieval.
4.  **intfloat/multilingual-e5-base**: Deep multilingual semantic support.
5.  **intfloat/multilingual-e5-small**: Lightweight multilingual support.
6.  **all-MiniLM-L6-v2**: Standard high-speed baseline.

## Configuration

All system behavior is controlled via JSON in `configs/`:
- `ingestion_*.json`: Controls vector dimensions, prefixes, and enrichment keys.
- `query_*.json`: Defines search types, filters, and similarity thresholds.
- `tests/configs/test_runtime_config.json`: Master switch for toggling between Sample and Production datasets.
