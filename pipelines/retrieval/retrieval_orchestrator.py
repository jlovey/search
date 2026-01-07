import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipelines.retrieval.models.product_retrievers import MiniLMRetriever, E5MLRetriever

def run_retrieval(config_path):
    with open(config_path, 'r') as f:
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
            if q_config["type"] == "hybrid":
                print(f" - [{r['product_id']}] {r['productName']} (RRF Score: {r['rrf_score']:.4f})")
            else:
                print(f" - [{r['original_id']}] {r['name']} (Score: {r['score']:.4f})")
        
        if not results:
            print(" - No results found.")
        print("-" * 30)

if __name__ == "__main__":
    config_file = "configs/query_minilm.json"
    run_retrieval(config_file)
