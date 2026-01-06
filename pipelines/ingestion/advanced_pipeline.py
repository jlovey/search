import json
import os
import sys
import uuid
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.services.qdrant_service import QdrantHelper
from models.embedding_service import EmbeddingService

def run_ingestion(data_path, config_path):
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 2. Load Data
    with open(data_path, 'r') as f:
        products = json.load(f)
    
    print(f"Loaded {len(products)} products from {data_path}")

    # 3. Initialize Services
    qdrant = QdrantHelper()
    embedder = EmbeddingService(config["dense_model"])

    # 4. Prepare Corpus for Sparse Model fitting
    all_text = []
    print("Preparing corpus for sparse model...")
    for p in products:
        all_text.append(EmbeddingService.flatten_content(p, config["sparse_keys"]))
    
    embedder.fit_sparse_model(all_text)
    embedder.save_sparse_model("models/sparse_model.pkl")

    # 5. Create Collection if not exists
    # We specify multiple vectors for the collection logic
    if not qdrant.collection_exists(config["collection_name"]):
        print(f"Creating collection {config['collection_name']}...")
        vector_config = {"size": config["vector_size"], "distance": "COSINE"}
        # Although we push separate docs, we define both slots in the schema for flexibility
        sparse_config = {"name": "sparse-vector"}
        qdrant.create_collection(config["collection_name"], vector_config, sparse_config)

    # 6. Process and Index
    print("Starting vectorization and indexing...")
    
    dense_points = []
    sparse_points = []

    for i, product in enumerate(tqdm(products)):
        # Extract product ID
        prod_id = str(product.get("id", i))
        
        # Prepare content strings
        dense_content = EmbeddingService.flatten_content(product, config["dense_keys"])
        sparse_content = EmbeddingService.flatten_content(product, config["sparse_keys"])
        
        # Generate Embeddings
        dense_prefix = config.get("dense_prefix", "")
        dense_vec = embedder.get_dense_embeddings([dense_content], prefix=dense_prefix)[0]
        sparse_vec = embedder.get_sparse_embeddings([sparse_content])[0]
        
        # Payload (Response keys)
        if config["response_keys"] == "all":
            payload = product.copy()
        else:
            payload = {k: product[k] for k in config["response_keys"] if k in product}
            
        # Move product id to original_id to maintain uniqueness across different vector types
        payload["original_id"] = prod_id

        # 1. Create Dense Document with unique deterministic UUID
        point_id_dense = str(uuid.uuid5(uuid.NAMESPACE_DNS, prod_id + "_dense"))
        dense_points.append({
            "id": point_id_dense,
            "vector": dense_vec,
            "payload": {**payload, "vector_type": "dense"}
        })

        # 2. Create Sparse Document with unique deterministic UUID
        point_id_sparse = str(uuid.uuid5(uuid.NAMESPACE_DNS, prod_id + "_sparse"))
        sparse_points.append({
            "id": point_id_sparse,
            "vector": {"sparse-vector": sparse_vec},
            "payload": {**payload, "vector_type": "sparse"}
        })

    # 7. Upload to Qdrant
    # Note: We need to use raw upload methods or extend QdrantHelper
    # I'll use the existing client inside QdrantHelper
    from qdrant_client.http import models as q_models
    
    print(f"Uploading {len(dense_points)} dense documents...")
    qdrant.client.upsert(
        collection_name=config["collection_name"],
        points=[
            q_models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
            for p in dense_points
        ]
    )

    print(f"Uploading {len(sparse_points)} sparse documents...")
    qdrant.client.upsert(
        collection_name=config["collection_name"],
        points=[
            q_models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
            for p in sparse_points
        ]
    )

    print("Ingestion complete.")

if __name__ == "__main__":
    # Example usage:
    # DATA must exist in data/products.json
    data_file = "data/sample_products.json" 
    conf_file = "configs/ingestion_minilm.json"
    
    if os.path.exists(data_file):
        run_ingestion(data_file, conf_file)
    else:
        print(f"Error: Data file {data_file} not found. please create it first.")
