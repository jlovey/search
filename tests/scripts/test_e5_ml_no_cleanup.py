import json
import pytest
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipelines.ingestion.ingestion_orchestrator import run_ingestion
from pipelines.retrieval.retrieval_orchestrator import run_retrieval
from pipelines.retrieval.cleanup_collection import run_cleanup

def load_e5_ml_config():
    config_path = "tests/configs/test_runtime_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    return [s for s in config["test_suites"] if s["name"] == "multilingual_e5"]

@pytest.mark.parametrize("suite", load_e5_ml_config())
def test_e5_ml_pipeline_no_cleanup(suite):
    print(f"\nRunning test suite (No Cleanup): {suite['name']}")
    
    ingestion_cfg = suite["ingestion_config"]
    query_cfg = suite["query_config"]
    data_path = suite["data_path"]

    # 1. Cleanup (ensure clean state before starting)
    print(f"Purging existing collection if any...")
    run_cleanup(ingestion_cfg)

    # 2. Ingestion
    print(f"Running ingestion with {ingestion_cfg}...")
    run_ingestion(data_path, ingestion_cfg)

    # 3. Retrieval
    print(f"Running retrieval with {query_cfg}...")
    run_retrieval(query_cfg)

    # Note: No final cleanup step here

    print(f"Test suite {suite['name']} (No Cleanup) passed successfully.")
