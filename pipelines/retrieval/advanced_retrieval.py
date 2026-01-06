import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.services.qdrant_service import QdrantHelper
from models.embedding_service import EmbeddingService
from qdrant_client.http import models as q_models

def reciprocal_rank_fusion(dense_hits, sparse_hits, limit=10):
    """Simple RRF to combine results from multiple retrieval types."""
    scores = {}
    
    # helper to get product ID (original ID or name) to avoid duplicate entries in results
    def get_item_id(hit):
        # We stored the original payload. If we want to merge, we use the product 'original_id' from payload
        return hit.payload.get("original_id")

    for rank, hit in enumerate(dense_hits):
        item_id = get_item_id(hit)
        scores[item_id] = scores.get(item_id, 0) + 1 / (60 + rank + 1)

    for rank, hit in enumerate(sparse_hits):
        item_id = get_item_id(hit)
        scores[item_id] = scores.get(item_id, 0) + 1 / (60 + rank + 1)

    # Sort by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return matched objects (reconstructing a bit for display)
    # In a real app, you'd fetch the full docs again or keep a map
    combined_results = []
    
    # Create a map for quick lookup from hits
    all_hits = {get_item_id(h): h.payload for h in dense_hits + sparse_hits}
    
    for item_id, score in sorted_scores[:limit]:
        combined_results.append({
            "product_id": item_id,
            "productName": all_hits[item_id].get("productName"),
            "rrf_score": score,
            "metadata": all_hits[item_id]
        })
        
    return combined_results

def run_retrieval(config_path):
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    collection_name = config["collection_name"]
    
    # 2. Initialize Services
    qdrant = QdrantHelper()
    embedder = EmbeddingService(config["dense_model"])
    embedder.load_sparse_model(config["sparse_model_path"])

    print(f"\n--- Running Retrieval Pipelines for Collection: {collection_name} ---\n")

    for q_config in config["queries"]:
        name = q_config["name"]
        query_text = q_config["query_text"]
        search_type = q_config["type"]
        limit = q_config.get("limit", 5)
        query_filter = q_config.get("filter")

        print(f"Executing: {name} (Type: {search_type})")
        print(f"Query: '{query_text}'")

        results = []

        if search_type == "dense":
            query_prefix = q_config.get("dense_prefix") or config.get("dense_prefix", "")
            vector = embedder.get_dense_embeddings([query_text], prefix=query_prefix)[0]
            # Filters for vector_type=dense only
            must_filters = [{"key": "vector_type", "match": {"value": "dense"}}]
            if query_filter:
                must_filters.extend(query_filter.get("must", []))
            
            hits = qdrant.client.query_points(
                collection_name=collection_name,
                query=vector,
                query_filter=q_models.Filter(must=must_filters),
                limit=limit
            ).points
            results = [{"id": h.id, "score": h.score, "name": h.payload.get("productName")} for h in hits]

        elif search_type == "sparse":
            sparse_data = embedder.get_sparse_embeddings([query_text])[0]
            
            must_filters = [{"key": "vector_type", "match": {"value": "sparse"}}]
            if query_filter:
                must_filters.extend(query_filter.get("must", []))

            hits = qdrant.client.query_points(
                collection_name=collection_name,
                query=q_models.SparseVector(
                    indices=sparse_data["indices"],
                    values=sparse_data["values"]
                ),
                using="sparse-vector",
                query_filter=q_models.Filter(must=must_filters),
                limit=limit
            ).points
            results = [{"id": h.id, "score": h.score, "name": h.payload.get("productName")} for h in hits]

        elif search_type == "hybrid":
            # 1. Search Dense
            query_prefix = q_config.get("dense_prefix") or config.get("dense_prefix", "")
            dense_vec = embedder.get_dense_embeddings([query_text], prefix=query_prefix)[0]
            dense_must = [{"key": "vector_type", "match": {"value": "dense"}}]
            if query_filter: dense_must.extend(query_filter.get("must", []))
            
            dense_hits = qdrant.client.query_points(
                collection_name=collection_name,
                query=dense_vec,
                query_filter=q_models.Filter(must=dense_must),
                limit=limit * 2
            ).points

            # 2. Search Sparse
            sparse_data = embedder.get_sparse_embeddings([query_text])[0]
            sparse_must = [{"key": "vector_type", "match": {"value": "sparse"}}]
            if query_filter: sparse_must.extend(query_filter.get("must", []))

            sparse_hits = qdrant.client.query_points(
                collection_name=collection_name,
                query=q_models.SparseVector(
                    indices=sparse_data["indices"],
                    values=sparse_data["values"]
                ),
                using="sparse-vector",
                query_filter=q_models.Filter(must=sparse_must),
                limit=limit * 2
            ).points

            # 3. Fuse
            results = reciprocal_rank_fusion(dense_hits, sparse_hits, limit=limit)

        # Print results
        for r in results:
            if search_type == "hybrid":
                 print(f" - [{r['product_id']}] {r['productName']} (RRF Score: {r['rrf_score']:.4f})")
            else:
                 print(f" - [{r['id']}] {r['name']} (Score: {r['score']:.4f})")
        
        if not results:
            print(" - No results found.")
        print("-" * 30)

if __name__ == "__main__":
    conf_file = "configs/query_minilm.json"
    run_retrieval(conf_file)
