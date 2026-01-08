import json
import pytest
import os
import sys
import time

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pipelines.ingestion.ingestion_orchestrator import run_ingestion
from pipelines.retrieval.retrieval_orchestrator import run_retrieval
from pipelines.retrieval.cleanup_collection import run_cleanup

def load_config():
    config_path = "tests/configs/test_runtime_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return [s for s in config["test_suites"] if s["name"] == "multilingual_e5_base"]

@pytest.mark.parametrize("suite", load_config())
def test_full_cycle(suite):
    print(f"\nRunning full cycle test: {{suite['name']}}")
    ingestion_cfg = suite["ingestion_config"]
    query_cfg = suite["query_config"]
    data_path = suite["data_path"]

    start_total = time.time()

    print("Step 1: Cleanup")
    run_cleanup(ingestion_cfg)
    
    print("Step 2: Ingestion")
    start_ingest = time.time()
    run_ingestion(data_path, ingestion_cfg)
    print(f"Ingestion took: {{time.time() - start_ingest:.2f}}s")
    
    print("Step 3: Retrieval")
    start_query = time.time()
    run_retrieval(query_cfg)
    print(f"Retrieval took: {{time.time() - start_query:.2f}}s")
    
    print("Step 4: Cleanup")
    run_cleanup(ingestion_cfg)
    
    print(f"\n--- Full Cycle for {{suite['name']}} completed in {{time.time() - start_total:.2f}s ---")
