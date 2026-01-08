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

    def run(self, data_path):
        json_files = []
        if os.path.isfile(data_path):
            json_files.append(data_path)
        elif os.path.isdir(data_path):
            for f in os.listdir(data_path):
                if f.endswith('.json'):
                    json_files.append(os.path.join(data_path, f))
        
        if not json_files:
            print(f"No JSON files found in {data_path}")
            return

        print(f"Processing {len(json_files)} files from {data_path}...")
        
        # Aggregate all products to fit sparse model and then prepare points
        all_products = []
        for file_path in json_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                products = json.load(f)
                if isinstance(products, list):
                    all_products.extend(products)
                else:
                    all_products.append(products)
        
        print(f"Loaded {len(all_products)} products total.")
        
        # Fit sparse model on the entire corpus
        print("Fitting sparse model on the entire corpus...")
        all_text = [EnrichmentService.enrich_sparse(p, self.config["sparse_keys"]) for p in all_products]
        self.embedder.fit_sparse_model(all_text)
        self.embedder.save_sparse_model("models/sparse_model.pkl")

        # Create Collection
        if not self.qdrant.collection_exists(self.config["collection_name"]):
            print(f"Creating collection: {self.config['collection_name']}")
            self.qdrant.create_collection(
                self.config["collection_name"], 
                {"size": self.config["vector_size"], "distance": "COSINE"},
                {"name": "sparse-vector"}
            )

        # Prepare points for all products
        dense_points, sparse_points = self.prepare_points(all_products)
        
        print(f"Uploading {len(dense_points)} dense and {len(sparse_points)} sparse points in batches...")
        self.qdrant.batch_upsert(self.config["collection_name"], dense_points)
        self.qdrant.batch_upsert(self.config["collection_name"], sparse_points)
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
