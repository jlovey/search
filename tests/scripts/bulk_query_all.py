import json
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipelines.retrieval.retrieval_orchestrator import run_retrieval

def bulk_query():
    config_path = "tests/configs/test_runtime_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    print(f"\n{'='*50}")
    print(f"STARTING BULK QUERY TESTING FOR {len(config['test_suites'])} MODELS")
    print(f"{'='*50}\n")

    for suite in config["test_suites"]:
        print(f"\n>>> Running Queries for Model: {suite['name']}")
        query_cfg = suite["query_config"]
        
        try:
            run_retrieval(query_cfg)
            print(f"Successfully tested queries for {suite['name']}")
        except Exception as e:
            print(f"Error testing {suite['name']}: {str(e)}")

    print(f"\n{'='*50}")
    print("BULK QUERY TESTING COMPLETE")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    bulk_query()
