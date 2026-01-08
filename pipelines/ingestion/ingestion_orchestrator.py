import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipelines.ingestion.models.product_ingestors import MiniLMIngestor, GenericPrefixIngestor

def run_ingestion(data_path, config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    model_name = config.get("dense_model", "")
    
    # Mapping of exact model names to their specialized ingestors
    INGESTOR_MAP = {
        "all-MiniLM-L6-v2": MiniLMIngestor,
        "intfloat/multilingual-e5-base": GenericPrefixIngestor,
        "intfloat/multilingual-e5-small": GenericPrefixIngestor,
        "BAAI/bge-small-en-v1.5": GenericPrefixIngestor,
        "mixedbread-ai/mxbai-embed-large-v1": GenericPrefixIngestor,
        "Qwen/Qwen3-Embedding-0.6B": GenericPrefixIngestor
    }

    ingestor_class = INGESTOR_MAP.get(model_name)
    
    if ingestor_class:
        ingestor = ingestor_class(config)
    else:
        # Default or Fallback
        print(f"Warning: No specialized ingestor for '{model_name}'. Using MiniLMIngestor as base.")
        ingestor = MiniLMIngestor(config)
    
    ingestor.run(data_path)

if __name__ == "__main__":
    data_path = "data/sample_products.json"
    config_file = "configs/ingestion_minilm.json"
    run_ingestion(data_path, config_file)
