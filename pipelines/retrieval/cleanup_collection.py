import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.services.qdrant_service import QdrantHelper

def run_cleanup(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    collection_name = config.get("collection_name")
    if not collection_name:
        print("Error: No collection_name found in config.")
        return

    qdrant = QdrantHelper()
    print(f"Attempting to delete collection: {collection_name}...")
    result = qdrant.delete_collection(collection_name)
    print(result)

if __name__ == "__main__":
    # Default to the advanced ingestion config
    conf_file = "configs/ingestion_minilm.json"
    run_cleanup(conf_file)
