import json
import pytest
import os
import sys
import time

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pipelines.ingestion.ingestion_orchestrator import run_ingestion
from pipelines.retrieval.cleanup_collection import run_cleanup

def load_config():
    config_path = "tests/configs/test_runtime_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return [s for s in config["test_suites"] if s["name"] == "standard_mini_lm"]

@pytest.mark.parametrize("suite", load_config())
def test_ingest_only(suite):
    print(f"\nRunning ingest-only test: {{suite['name']}}")
    ingestion_cfg = suite["ingestion_config"]
    data_path = suite["data_path"]

    run_cleanup(ingestion_cfg)
    
    start_time = time.time()
    run_ingestion(data_path, ingestion_cfg)
    print(f"Ingestion for {{suite['name']}} took: {{time.time() - start_time:.2f}}s")
