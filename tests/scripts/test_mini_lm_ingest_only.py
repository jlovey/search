import json
import pytest
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipelines.ingestion.ingestion_orchestrator import run_ingestion
from pipelines.retrieval.cleanup_collection import run_cleanup

def load_mini_lm_config():
    config_path = "tests/configs/test_runtime_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return [s for s in config["test_suites"] if s["name"] == "standard_mini_lm"]

@pytest.mark.parametrize("suite", load_mini_lm_config())
def test_mini_lm_ingest_only(suite):
    print(f"\nRunning Ingest-Only test suite: {suite['name']}")
    
    ingestion_cfg = suite["ingestion_config"]
    data_path = suite["data_path"]

    # 1. Cleanup (ensure clean state)
    print(f"Purging existing collection if any...")
    run_cleanup(ingestion_cfg)

    # 2. Ingestion
    print(f"Running ingestion with {ingestion_cfg}...")
    run_ingestion(data_path, ingestion_cfg)

    print(f"Ingest-Only test for {suite['name']} passed successfully.")
