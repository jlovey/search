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
    
    # Structure: results_by_type[type][query_text][model_name] = results
    results_by_type = {}
    model_stats = {}
    model_names = [suite["name"] for suite in config["test_suites"]]

    for suite in config["test_suites"]:
        model_name = suite["name"]
        print(f"\n>>> Running Queries for Model: {model_name}")
        query_cfg_path = suite["query_config"]
        
        try:
            start_time = time.time()
            suite_results = run_retrieval(query_cfg_path)
            duration = time.time() - start_time
            
            num_queries = len(suite_results)
            avg_time = duration / num_queries if num_queries > 0 else 0
            
            model_stats[model_name] = {
                "duration": duration,
                "avg": avg_time,
                "count": num_queries
            }

            for qr in suite_results:
                q_type = qr["type"].upper()
                q_text = qr["query_text"]
                
                if q_type not in results_by_type:
                    results_by_type[q_type] = {}
                if q_text not in results_by_type[q_type]:
                    results_by_type[q_type][q_text] = {}
                
                results_by_type[q_type][q_text][model_name] = qr["results"]

            print(f"Successfully tested {num_queries} queries for {model_name} in {duration:.2f}s")
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")

    # Create results directory if it doesn't exist
    os.makedirs("tests/results", exist_ok=True)

    # 1. Generate Model Comparison Report
    comparison_path = "tests/results/model_comparison.md"
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write("# Model Comparison Report\n\n")
        
        # Sort types so DENSE and SPARSE come first if they exist
        types = sorted(results_by_type.keys())
        
        for q_type in types:
            f.write(f"## Search Type: {q_type}\n\n")
            
            # Header
            header = "| Query | " + " | ".join(model_names) + " |"
            sep = "| :--- | " + " | ".join(["---"] * len(model_names)) + " |"
            f.write(header + "\n")
            f.write(sep + "\n")
            
            # Rows (one per query_text)
            queries = sorted(results_by_type[q_type].keys())
            for q_text in queries:
                row = f"| **{q_text}** |"
                for m_name in model_names:
                    m_results = results_by_type[q_type][q_text].get(m_name, [])
                    # Take top 3
                    top_3 = m_results[:3]
                    res_strings = []
                    for i, r in enumerate(top_3):
                        res_strings.append(f"{i+1}. [{r['product_id']}] {r['product_name']}<br>(S:{r['score']:.3f}, R:{r['raw_score']:.3f})")
                    
                    cell_content = "<br>".join(res_strings) if res_strings else "N/A"
                    row += f" {cell_content} |"
                f.write(row + "\n")
            f.write("\n")

    # 2. Generate Performance Benchmarks Report
    perf_path = "tests/results/performance_benchmarks.md"
    with open(perf_path, "w", encoding="utf-8") as f:
        # Calculate total queries across all models for the title n
        total_q_count = sum(s["count"] for s in model_stats.values())
        # The user wants "n Queries x 4 Types" in the title. 
        # Here n usually means unique query texts.
        unique_queries = set()
        for t in results_by_type:
            for q in results_by_type[t]:
                unique_queries.add(q)
        n = len(unique_queries)
        
        f.write(f"# Performance Benchmarking Summary ({n} Queries x 4 Types)\n\n")
        f.write("| Model | Total Suite Time | Avg Latency / Query | Query Count |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        # Sort models by duration or just keep order
        for m_name in model_names:
            if m_name in model_stats:
                s = model_stats[m_name]
                f.write(f"| **{m_name}** | {s['duration']:.2f}s | {s['avg']:.4f}s | {s['count']} |\n")
    
    print(f"\n{'='*50}")
    print("BULK QUERY TESTING COMPLETE")
    print(f"Reports generated in tests/results/")
    print(f"- {comparison_path}")
    print(f"- {perf_path}")
    print(f"Total Execution Time: {time.time() - overall_start:.2f}s")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    bulk_query()
