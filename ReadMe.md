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
├── configs/                     # Centralized Model & Query Configurations
│   ├── ingestion_minilm.json    # MiniLM Settings
│   ├── ingestion_e5_small.json  # E5 Small Settings
│   ├── ingestion_bge_small.json # BGE Small Settings
│   ├── ingestion_mxbai_large.json # MXBAI Settings
│   ├── ingestion_qwen3.json     # Qwen3 Settings
│   └── ...                      # Corresponding query configs
├── pipelines/                   # Core Logic & Orchestration
│   ├── ingestion/               
│   │   ├── models/              # Model-specific Indexing Logic
│   │   └── ingestion_orchestrator.py
│   └── retrieval/               
│       ├── models/              # Model-specific Search Logic
│       └── retrieval_orchestrator.py
│       └── cleanup_collection.py     # Utility to purge Qdrant collections
├── src/                         # Shared Application Core
│   ├── services/                
│   │   ├── qdrant_service.py    # Low-level Qdrant Client Wrapper
│   │   ├── enrichment_service.py # Markdown & Flattening Logic
│   │   └── metrics_service.py   # Accuracy & Evaluation Metrics
│   └── schema/                  # Pydantic Data Models
├── models/                      # ML & Embedding logic
│   ├── embedding_service.py     # Shared Vectorization logic (MiniLM, E5, TF-IDF)
│   └── sparse_model.pkl         # Persisted TF-IDF vocabulary
├── tests/                       # Automated Testing Suite
│   ├── scripts/                 # Pytest scenarios (Full Cycle vs No-Cleanup)
│   └── configs/                 # Master Test runtime configuration
├── data/                        # Sample product datasets (JSON)
├── ReadMe.md                    # Project Documentation
└── requirements.txt             # Python Dependencies
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

### Testing

Tests are now organized by model folders. Each folder contains three standardized test scripts:
- `test_full_cycle.py`: Purge → Index → Search → Cleanup.
- `test_ingest_only.py`: Purge → Index.
- `test_query_only.py`: Search existing collection.

To run tests for a specific model (e.g., Qwen3):
```bash
pytest -v -s tests/scripts/qwen3/test_full_cycle.py
```

### Supported Models
The system is pre-configured for:
1.  **Qwen3-Embedding-0.6B**: High-performance semantic model with 32k context.
2.  **BAAI/bge-small-en-v1.5**: Best-in-class small English model.
3.  **mixedbread-ai/mxbai-embed-large-v1**: State-of-the-art accuracy.
4.  **intfloat/multilingual-e5-small**: Lightweight multilingual support.
5.  **all-MiniLM-L6-v2**: Standard baseline for performance.

## Configuration
- `configs/ingestion_minilm.json`: Ingestion config for MiniLM model.
- `configs/query_minilm.json`: Query config for MiniLM model.
- `configs/ingestion_e5_ml.json`: Ingestion config for Multilingual E5 model.
- `configs/query_e5_ml.json`: Query config for Multilingual E5 model.
- `tests/configs/test_runtime_config.json`: Master configuration for automated tests.
