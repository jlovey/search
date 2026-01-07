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
        
        return [{
            "product_id": pid,
            "productName": all_payloads[pid].get("productName"),
            "rrf_score": score,
            "metadata": all_payloads[pid]
        } for pid, score in sorted_scores[:limit]]

    def search(self, q_config):
        raise NotImplementedError("Subclasses must implement search")

class MiniLMRetriever(BaseRetriever):
    def search(self, q_config):
        return self._execute(q_config, prefix="")

    def _execute(self, q_config, prefix):
        query_text = q_config["query_text"]
        limit = q_config.get("limit", 5)
        search_type = q_config["type"]
        col = self.config["collection_name"]

        if search_type == "dense":
            vec = self.embedder.get_dense_embeddings([query_text], prefix=prefix)[0]
            hits = self.qdrant.client.query_points(
                collection_name=col, query=vec, 
                query_filter=q_models.Filter(must=[{"key": "vector_type", "match": {"value": "dense"}}]),
                limit=limit
            ).points
            return [{"id": h.id, "score": h.score, "name": h.payload.get("productName"), "original_id": h.payload.get("original_id")} for h in hits]

        elif search_type == "sparse":
            sparse_data = self.embedder.get_sparse_embeddings([query_text])[0]
            hits = self.qdrant.client.query_points(
                collection_name=col,
                query=q_models.SparseVector(indices=sparse_data["indices"], values=sparse_data["values"]),
                using="sparse-vector",
                query_filter=q_models.Filter(must=[{"key": "vector_type", "match": {"value": "sparse"}}]),
                limit=limit
            ).points
            return [{"id": h.id, "score": h.score, "name": h.payload.get("productName"), "original_id": h.payload.get("original_id")} for h in hits]

        elif search_type == "hybrid":
            dense_hits = self.qdrant.client.query_points(col, query=self.embedder.get_dense_embeddings([query_text], prefix=prefix)[0], 
                                                        query_filter=q_models.Filter(must=[{"key":"vector_type", "match":{"value":"dense"}}]), limit=limit*2).points
            sparse_data = self.embedder.get_sparse_embeddings([query_text])[0]
            sparse_hits = self.qdrant.client.query_points(col, query=q_models.SparseVector(indices=sparse_data["indices"], values=sparse_data["values"]),
                                                         using="sparse-vector", query_filter=q_models.Filter(must=[{"key":"vector_type", "match":{"value":"sparse"}}]), limit=limit*2).points
            # Potential specializing for MiniLM here
            return self.reciprocal_rank_fusion(dense_hits, sparse_hits, limit=limit)

class E5MLRetriever(BaseRetriever):
    def search(self, q_config):
        prefix = q_config.get("dense_prefix") or self.config.get("dense_prefix", "query: ")
        return self._execute(q_config, prefix=prefix)

    def _execute(self, q_config, prefix):
        query_text = q_config["query_text"]
        limit = q_config.get("limit", 5)
        search_type = q_config["type"]
        col = self.config["collection_name"]

        if search_type == "dense":
            vec = self.embedder.get_dense_embeddings([query_text], prefix=prefix)[0]
            hits = self.qdrant.client.query_points(col, query=vec, query_filter=q_models.Filter(must=[{"key":"vector_type", "match":{"value":"dense"}}]), limit=limit).points
            return [{"id": h.id, "score": h.score, "name": h.payload.get("productName"), "original_id": h.payload.get("original_id")} for h in hits]

        elif search_type == "sparse":
            sparse_data = self.embedder.get_sparse_embeddings([query_text])[0]
            hits = self.qdrant.client.query_points(col, query=q_models.SparseVector(indices=sparse_data["indices"], values=sparse_data["values"]),
                                                 using="sparse-vector", query_filter=q_models.Filter(must=[{"key":"vector_type", "match":{"value":"sparse"}}]), limit=limit).points
            return [{"id": h.id, "score": h.score, "name": h.payload.get("productName"), "original_id": h.payload.get("original_id")} for h in hits]

        elif search_type == "hybrid":
            dense_hits = self.qdrant.client.query_points(col, query=self.embedder.get_dense_embeddings([query_text], prefix=prefix)[0], 
                                                        query_filter=q_models.Filter(must=[{"key":"vector_type", "match":{"value":"dense"}}]), limit=limit*2).points
            sparse_data = self.embedder.get_sparse_embeddings([query_text])[0]
            sparse_hits = self.qdrant.client.query_points(col, query=q_models.SparseVector(indices=sparse_data["indices"], values=sparse_data["values"]),
                                                         using="sparse-vector", query_filter=q_models.Filter(must=[{"key":"vector_type", "match":{"value":"sparse"}}]), limit=limit*2).points
            # Potential specializing for E5-ML here (e.g. better reranking)
            return self.reciprocal_rank_fusion(dense_hits, sparse_hits, limit=limit)
