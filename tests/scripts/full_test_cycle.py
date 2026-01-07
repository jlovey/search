import json
import pytest
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipelines.ingestion.ingestion_orchestrator import run_ingestion
from pipelines.retrieval.retrieval_orchestrator import run_retrieval
from pipelines.retrieval.cleanup_collection import run_cleanup

def load_test_config():
    config_path = "tests/configs/test_runtime_config.json"
    with open(config_path, "r") as f:
        return json.load(f)

@pytest.mark.parametrize("suite", load_test_config()["test_suites"])
def test_full_pipeline(suite):
    print(f"\nRunning test suite: {suite['name']}")
    
    ingestion_cfg = suite["ingestion_config"]
    query_cfg = suite["query_config"]
    data_file = suite["data_file"]

    # 1. Cleanup (ensure clean state)
    print(f"Purging existing collection if any...")
    run_cleanup(ingestion_cfg)

    # 2. Ingestion
    print(f"Running ingestion with {ingestion_cfg}...")
    run_ingestion(data_file, ingestion_cfg)

    # 3. Retrieval
    print(f"Running retrieval with {query_cfg}...")
    # Capturing stdout isn't strictly necessary for the test logic but we want to ensure it doesn't crash
    run_retrieval(query_cfg)

    # 4. Cleanup (leave the environment clean)
    print(f"Cleaning up collection...")
    run_cleanup(ingestion_cfg)

    print(f"Test suite {suite['name']} passed successfully.")
