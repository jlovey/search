import json
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipelines.retrieval.retrieval_orchestrator import run_retrieval

import time

def bulk_query():
    config_path = "tests/configs/test_runtime_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    print(f"\n{'='*50}")
    print(f"STARTING BULK QUERY TESTING FOR {len(config['test_suites'])} MODELS")
    print(f"{'='*50}\n")

    overall_start = time.time()
    stats = []

    for suite in config["test_suites"]:
        print(f"\n>>> Running Queries for Model: {suite['name']}")
        query_cfg_path = suite["query_config"]
        
        # Load query config to count queries
        with open(query_cfg_path, "r", encoding="utf-8") as f:
            query_cfg = json.load(f)
        num_queries = len(query_cfg.get("queries", []))
        
        try:
            start_time = time.time()
            run_retrieval(query_cfg_path)
            duration = time.time() - start_time
            avg_time = duration / num_queries if num_queries > 0 else 0
            
            stats.append({
                "model": suite["name"], 
                "duration": duration, 
                "avg": avg_time, 
                "count": num_queries
            })
            print(f"Successfully tested {num_queries} queries for {suite['name']} in {duration:.2f}s (Avg: {avg_time:.4f}s/query)")
        except Exception as e:
            print(f"Error testing {suite['name']}: {str(e)}")

    print(f"\n{'='*50}")
    print("BULK QUERY TESTING COMPLETE")
    print(f"Total Time: {time.time() - overall_start:.2f}s")
    print("-" * 60)
    print(f"{'Model':25} | {'Suite Time':10} | {'Avg/Query':10} | {'Count'}")
    print("-" * 60)
    for s in stats:
        print(f"{s['model']:25} | {s['duration']:9.2f}s | {s['avg']:9.4f}s | {s['count']}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    bulk_query()
