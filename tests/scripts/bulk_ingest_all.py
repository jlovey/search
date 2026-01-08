import json
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipelines.ingestion.ingestion_orchestrator import run_ingestion
from pipelines.retrieval.cleanup_collection import run_cleanup

import time

def bulk_ingest():
    config_path = "tests/configs/test_runtime_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    print(f"\n{'='*50}")
    print(f"STARTING BULK INGESTION FOR {len(config['test_suites'])} MODELS")
    print(f"{'='*50}\n")

    overall_start = time.time()
    stats = []

    for suite in config["test_suites"]:
        print(f"\n>>> Processing Model: {suite['name']}")
        ingestion_cfg = suite["ingestion_config"]
        data_path = suite["data_path"]

        print(f"Cleaning up collection if exists...")
        run_cleanup(ingestion_cfg)
        
        print(f"Running ingestion from: {data_path} using {ingestion_cfg}")
        start_time = time.time()
        run_ingestion(data_path, ingestion_cfg)
        duration = time.time() - start_time
        
        stats.append({"model": suite["name"], "duration": duration})
        print(f"Successfully ingested {suite['name']} in {duration:.2f}s")

    print(f"\n{'='*50}")
    print("BULK INGESTION COMPLETE")
    print(f"Total Time: {time.time() - overall_start:.2f}s")
    print("-" * 30)
    for s in stats:
        print(f" - {s['model']:25} | {s['duration']:.2f}s")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    bulk_ingest()
