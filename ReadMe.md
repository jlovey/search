# Advanced Product Search with Qdrant

This project implements a professional, full-fledged search application using Qdrant. It supports hybrid search (Dense + Sparse) with automated vectorization and RRF (Reciprocal Rank Fusion).

## Features
- **Hybrid Search**: Combines semantic understanding (Dense) with keyword matching (Sparse).
- **Automated Embedding**: Uses `sentence-transformers` for dense vectors and `TF-IDF` for sparse vectors.
- **Config-Driven Ingestion**: Define which fields to embed and filter via JSON.
- **Flexible Retrieval**: Supports pure dense, pure sparse, and hybrid search modes.
- **RRF Fusion**: Intelligently merges results from different retrieval methods.

## Project Structure

```text
search/
├── configs/                     # Centralized Model & Query Configurations
│   ├── ingestion_minilm.json    # MiniLM Specific Ingestion Settings
│   ├── query_minilm.json        # MiniLM Specific Query Scenarios (Dense, Sparse, Hybrid)
│   ├── ingestion_e5_ml.json     # Multilingual E5 Ingestion Settings
│   ├── query_e5_ml.json         # Multilingual E5 Query Scenarios
│   └── templates/               # Reusable config templates for new models
├── pipelines/                   # Core Logic & Orchestration
│   ├── ingestion/               
│   │   ├── models/              # Model-specific Indexing Logic
│   │   │   └── product_ingestors.py
│   │   └── ingestion_orchestrator.py # Entry point for data indexing
│   └── retrieval/               
│       ├── models/              # Model-specific Search & Reranking Logic
│       │   └── product_retrievers.py
│       ├── retrieval_orchestrator.py # Entry point for search execution
│       └── cleanup_collection.py     # Utility to purge Qdrant collections
├── src/                         # Shared Application Core
│   ├── services/                
│   │   ├── qdrant_service.py    # Low-level Qdrant Client Wrapper
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

### 4. Testing
We provide 5 dedicated test scenarios to verify different parts of the pipeline using `pytest`.

#### Run All Suites
Execute the complete lifecycle for all configured models (MiniLM and E5).
```bash
pytest -v -s tests/scripts/full_test_cycle.py
```

#### Model-Specific Full Cycles
Verify a full scenario (Purge → Index → Search → Cleanup) for a single model.
```bash
# MiniLM + BM25
pytest -v -s tests/scripts/test_mini_lm_full_cycle.py

# Multilingual E5 + BM25
pytest -v -s tests/scripts/test_e5_ml_full_cycle.py
```

#### Persistent Tests (No Cleanup)
Index and Query but skip the final deletion. Use this to inspect the data in Qdrant afterward.
```bash
# Keep MiniLM data
pytest -v -s tests/scripts/test_mini_lm_no_cleanup.py

# Keep E5 data
pytest -v -s tests/scripts/test_e5_ml_no_cleanup.py
```

## Configuration
- `configs/ingestion_minilm.json`: Ingestion config for MiniLM model.
- `configs/query_minilm.json`: Query config for MiniLM model.
- `configs/ingestion_e5_ml.json`: Ingestion config for Multilingual E5 model.
- `configs/query_e5_ml.json`: Query config for Multilingual E5 model.
- `tests/configs/test_runtime_config.json`: Master configuration for automated tests.
