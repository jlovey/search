import hashlib
import json
import os
import sys
import uuid
from tqdm import tqdm
from qdrant_client.http import models as q_models

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.services.qdrant_service import QdrantHelper
from src.services.enrichment_service import EnrichmentService
from models.embedding_service import EmbeddingService

class BaseIngestor:
    def __init__(self, config):
        self.config = config
        self.qdrant = QdrantHelper()
        self.embedder = EmbeddingService(config["dense_model"])
    
    @staticmethod
    def get_numeric_id(pid, suffix):
        # Create a stable 64-bit integer from the string
        identifier = f"{pid}_{suffix}"
        return int(hashlib.md5(identifier.encode()).hexdigest()[:15], 16)

    def prepare_points(self, products):
        raise NotImplementedError("Subclasses must implement prepare_points")

    def run(self, data_file):
        with open(data_file, 'r') as f:
            products = json.load(f)
        
        print(f"Loaded {len(products)} products from {data_file}")
        
        # Fit sparse model
        all_text = [EnrichmentService.enrich_sparse(p, self.config["sparse_keys"]) for p in products]
        self.embedder.fit_sparse_model(all_text)
        self.embedder.save_sparse_model("models/sparse_model.pkl")

        # Create Collection
        if not self.qdrant.collection_exists(self.config["collection_name"]):
            self.qdrant.create_collection(
                self.config["collection_name"], 
                {"size": self.config["vector_size"], "distance": "COSINE"},
                {"name": "sparse-vector"}
            )

        dense_points, sparse_points = self.prepare_points(products)
        
        print(f"Uploading {len(dense_points)} dense and {len(sparse_points)} sparse documents...")
        self.qdrant.client.upsert(self.config["collection_name"], points=dense_points)
        self.qdrant.client.upsert(self.config["collection_name"], points=sparse_points)
        print("Ingestion complete.")

class MiniLMIngestor(BaseIngestor):
    def prepare_points(self, products):
        dense_points = []
        sparse_points = []
        for i, product in enumerate(tqdm(products, desc="Vectorizing MiniLM")):
            pid = str(product.get("id", i))
            dense_content = EnrichmentService.enrich_dense(product, self.config["dense_keys"])
            sparse_content = EnrichmentService.enrich_sparse(product, self.config["sparse_keys"])
            
            # MiniLM logic: No prefix
            dense_vec = self.embedder.get_dense_embeddings([dense_content])[0]
            sparse_vec = self.embedder.get_sparse_embeddings([sparse_content])[0]
            
            payload = product.copy()
            payload["original_id"] = pid
            
            dense_points.append(q_models.PointStruct(
                id=self.get_numeric_id(pid, "dense"),
                vector=dense_vec,
                payload={**payload, "vector_type": "dense"}
            ))
            sparse_points.append(q_models.PointStruct(
                id=self.get_numeric_id(pid, "sparse"),
                vector={"sparse-vector": sparse_vec},
                payload={**payload, "vector_type": "sparse"}
            ))
        return dense_points, sparse_points

class E5MLIngestor(BaseIngestor):
    def prepare_points(self, products):
        dense_points = []
        sparse_points = []
        prefix = self.config.get("dense_prefix", "passage: ")
        for i, product in enumerate(tqdm(products, desc="Vectorizing E5-ML")):
            pid = str(product.get("id", i))
            dense_content = EnrichmentService.enrich_dense(product, self.config["dense_keys"])
            sparse_content = EnrichmentService.enrich_sparse(product, self.config["sparse_keys"])
            
            # E5 Logic: Use prefix
            dense_vec = self.embedder.get_dense_embeddings([dense_content], prefix=prefix)[0]
            sparse_vec = self.embedder.get_sparse_embeddings([sparse_content])[0]
            
            payload = product.copy()
            payload["original_id"] = pid
            
            dense_points.append(q_models.PointStruct(
                id=self.get_numeric_id(pid, "dense"),
                vector=dense_vec,
                payload={**payload, "vector_type": "dense"}
            ))
            sparse_points.append(q_models.PointStruct(
                id=self.get_numeric_id(pid, "sparse"),
                vector={"sparse-vector": sparse_vec},
                payload={**payload, "vector_type": "sparse"}
            ))
        return dense_points, sparse_points
