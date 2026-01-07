import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipelines.ingestion.models.product_ingestors import MiniLMIngestor, E5MLIngestor

def run_ingestion(data_file, config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_name = config.get("dense_model", "")
    
    # Mapping of exact model names to their specialized ingestors
    INGESTOR_MAP = {
        "intfloat/multilingual-e5-base": E5MLIngestor,
        "all-MiniLM-L6-v2": MiniLMIngestor
    }

    ingestor_class = INGESTOR_MAP.get(model_name)
    
    if ingestor_class:
        ingestor = ingestor_class(config)
    else:
        # Default or Fallback
        print(f"Warning: No specialized ingestor for '{model_name}'. Using MiniLMIngestor as base.")
        ingestor = MiniLMIngestor(config)
    
    ingestor.run(data_file)

if __name__ == "__main__":
    data_file = "data/sample_products.json"
    config_file = "configs/ingestion_minilm.json"
    run_ingestion(data_file, config_file)
