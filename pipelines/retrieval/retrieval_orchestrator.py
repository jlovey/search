import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipelines.retrieval.models.product_retrievers import MiniLMRetriever, E5MLRetriever

def run_retrieval(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    model_name = config.get("dense_model", "")
    
    # Mapping of exact model names to their specialized retrievers
    RETRIEVER_MAP = {
        "intfloat/multilingual-e5-base": E5MLRetriever,
        "all-MiniLM-L6-v2": MiniLMRetriever
    }

    retriever_class = RETRIEVER_MAP.get(model_name)
    
    if retriever_class:
        retriever = retriever_class(config)
    else:
        print(f"Warning: No specialized retriever for '{model_name}'. Using MiniLMRetriever as base.")
        retriever = MiniLMRetriever(config)
    
    print(f"\n--- Running Retrieval Pipelines for Collection: {config['collection_name']} ---")
    
    for q_config in config["queries"]:
        print(f"Executing: {q_config['name']} (Type: {q_config['type']})")
        results = retriever.search(q_config)
        
        for r in results:
            # Shared identifiers
            pid = r.get('product_id') or r.get('original_id')
            pname = r.get('productName') or r.get('name')
            score = r['score']
            raw = r['raw_score']
            
            label = "RRF" if q_config["type"] == "hybrid" else "Sim"
            print(f" - [{pid}] {pname} (Score: {score:.4f}, Raw {label}: {raw:.4f})")
        
        if not results:
            print(" - No results found.")
        print("-" * 30)

if __name__ == "__main__":
    config_file = "configs/query_minilm.json"
    run_retrieval(config_file)
