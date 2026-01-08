import json
import pytest
import os
import sys

# Add project root to sys.path
# Script is at tests/scripts/[model]/test_full_cycle.py -> 4 levels deep
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pipelines.ingestion.ingestion_orchestrator import run_ingestion
from pipelines.retrieval.retrieval_orchestrator import run_retrieval
from pipelines.retrieval.cleanup_collection import run_cleanup

def load_config():
    config_path = "tests/configs/test_runtime_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return [s for s in config["test_suites"] if s["name"] == "multilingual_e5_small"]

@pytest.mark.parametrize("suite", load_config())
def test_full_cycle(suite):
    print(f"\nRunning full cycle test: {{suite[name]}}")
    ingestion_cfg = suite["ingestion_config"]
    query_cfg = suite["query_config"]
    data_path = suite["data_path"]

    run_cleanup(ingestion_cfg)
    run_ingestion(data_path, ingestion_cfg)
    run_retrieval(query_cfg)
    run_cleanup(ingestion_cfg)
