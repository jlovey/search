import json
import pytest
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipelines.retrieval.retrieval_orchestrator import run_retrieval

def load_e5_config():
    config_path = "tests/configs/test_runtime_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return [s for s in config["test_suites"] if s["name"] == "multilingual_e5"]

@pytest.mark.parametrize("suite", load_e5_config())
def test_e5_ml_query_only(suite):
    print(f"\nRunning Query-Only test suite: {suite['name']}")
    
    query_cfg = suite["query_config"]

    # Retrieval only
    print(f"Running retrieval with {query_cfg}...")
    run_retrieval(query_cfg)

    print(f"Query-Only test for {suite['name']} passed successfully.")
