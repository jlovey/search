import json
import pytest
import os
import sys
import time

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pipelines.retrieval.retrieval_orchestrator import run_retrieval

def load_config():
    config_path = "tests/configs/test_runtime_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return [s for s in config["test_suites"] if s["name"] == "standard_mini_lm"]

@pytest.mark.parametrize("suite", load_config())
def test_query_only(suite):
    print(f"\nRunning query-only test: {{suite['name']}}")
    query_cfg = suite["query_config"]
    
    start_time = time.time()
    run_retrieval(query_cfg)
    print(f"Queries for {{suite['name']}} took: {{time.time() - start_time:.2f}}s")
