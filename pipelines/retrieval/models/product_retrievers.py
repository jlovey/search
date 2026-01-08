import json
import os
import sys
from qdrant_client.http import models as q_models

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.services.qdrant_service import QdrantHelper
from models.embedding_service import EmbeddingService

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.qdrant = QdrantHelper()
        self.embedder = EmbeddingService(config["dense_model"])
        self.embedder.load_sparse_model(config["sparse_model_path"])

    def reciprocal_rank_fusion(self, dense_hits, sparse_hits, limit=10):
        scores = {}
        # Max possible RRF score for 2 streams is (1/61 + 1/61) = 0.032786
        # We use this to normalize RRF to [0, 1]
        MAX_RRF_SINGLE = 1 / 61.0
        MAX_RRF_TOTAL = MAX_RRF_SINGLE * 2

        def get_item_id(hit):
            return hit.payload.get("original_id")

        for rank, hit in enumerate(dense_hits):
            idx = get_item_id(hit)
            scores[idx] = scores.get(idx, 0) + 1 / (60 + rank + 1)

        for rank, hit in enumerate(sparse_hits):
            idx = get_item_id(hit)
            scores[idx] = scores.get(idx, 0) + 1 / (60 + rank + 1)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        all_payloads = {get_item_id(h): h.payload for h in dense_hits + sparse_hits}
        
        results = []
        for pid, score in sorted_scores[:limit]:
            norm_score = min(1.0, score / MAX_RRF_TOTAL)
            results.append({
                "product_id": pid,
                "productName": all_payloads[pid].get("productName"),
                "raw_score": score,
                "score": norm_score, # Normalized score
                "metadata": all_payloads[pid]
            })
        return results

    def _normalize_hits(self, hits):
        """Helper to normalize raw scores of a hit list to [0, 1] range."""
        if not hits:
            return []
        
        raw_scores = [h.score for h in hits]
        # If all scores are the same, return 1.0 for all
        if len(raw_scores) == 1 or max(raw_scores) == min(raw_scores):
            return [{"id": h.id, "score": 1.0, "raw_score": h.score, "name": h.payload.get("productName"), "original_id": h.payload.get("original_id")} for h in hits]
        
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        
        results = []
        for h in hits:
            norm_score = (h.score - min_s) / (max_s - min_s)
            results.append({
                "id": h.id,
                "score": round(norm_score, 4),
                "raw_score": h.score,
                "name": h.payload.get("productName"),
                "original_id": h.payload.get("original_id")
            })
        return results

    def search(self, q_config):
        """Standard search implementation that handles dense, sparse, and hybrid."""
        query_text = q_config["query_text"]
        limit = q_config.get("limit", 5)
        search_type = q_config["type"]
        score_threshold = q_config.get("score_threshold")
        col = self.config["collection_name"]
        
        # Get prefix if it's an E5 style model
        prefix = q_config.get("dense_prefix") or self.config.get("dense_prefix", "")

        # Prepare Filters
        must_filters = [{"key": "vector_type", "match": {"value": "dense" if search_type == "dense" else "sparse"}}]
        if q_config.get("filter"):
            must_filters.extend(q_config["filter"].get("must", []))
        
        q_filter = q_models.Filter(must=must_filters)

        if search_type == "dense":
            vec = self.embedder.get_dense_embeddings([query_text], prefix=prefix)[0]
            hits = self.qdrant.client.query_points(
                collection_name=col, 
                query=vec, 
                query_filter=q_filter,
                score_threshold=score_threshold,
                limit=limit
            ).points
            return self._normalize_hits(hits)

        elif search_type == "sparse":
            sparse_data = self.embedder.get_sparse_embeddings([query_text])[0]
            hits = self.qdrant.client.query_points(
                collection_name=col,
                query=q_models.SparseVector(indices=sparse_data["indices"], values=sparse_data["values"]),
                using="sparse-vector",
                query_filter=q_filter,
                score_threshold=score_threshold,
                limit=limit
            ).points
            return self._normalize_hits(hits)

        elif search_type == "hybrid":
            # 1. Dense Search
            dense_vec = self.embedder.get_dense_embeddings([query_text], prefix=prefix)[0]
            dense_filter = q_models.Filter(must=[{"key": "vector_type", "match": {"value": "dense"}}] + (q_config.get("filter", {}).get("must", [])))
            dense_hits = self.qdrant.client.query_points(
                col, query=dense_vec, query_filter=dense_filter, score_threshold=score_threshold, limit=limit*2
            ).points

            # 2. Sparse Search
            sparse_data = self.embedder.get_sparse_embeddings([query_text])[0]
            sparse_filter = q_models.Filter(must=[{"key": "vector_type", "match": {"value": "sparse"}}] + (q_config.get("filter", {}).get("must", [])))
            sparse_hits = self.qdrant.client.query_points(
                col, 
                query=q_models.SparseVector(indices=sparse_data["indices"], values=sparse_data["values"]),
                using="sparse-vector", 
                query_filter=sparse_filter, 
                score_threshold=score_threshold,
                limit=limit*2
            ).points
            
            return self.reciprocal_rank_fusion(dense_hits, sparse_hits, limit=limit)

class GenericRetriever(BaseRetriever):
    """
    A generic retriever that handles dense, sparse, and hybrid search.
    Logic is inherited from BaseRetriever.
    """
    pass
