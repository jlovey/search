import json
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

class QdrantHelper:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)

    def collection_exists(self, collection_name):
        collections = self.client.get_collections().collections
        return any(c.name == collection_name for c in collections)

    def create_collection(self, collection_name, vector_config, sparse_config=None):
        if self.collection_exists(collection_name):
            return f"Collection '{collection_name}' already exists."
        
        # Dense vector configuration
        vectors_config = models.VectorParams(
            size=vector_config.get("size", 1536),
            distance=getattr(models.Distance, vector_config.get("distance", "COSINE"))
        )
        
        # Sparse vector configuration (optional)
        sparse_vectors_config = None
        if sparse_config:
            sparse_vectors_config = {
                sparse_config.get("name", "sparse-vector"): models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=sparse_config.get("on_disk", False)
                    )
                )
            }

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config
        )
        return f"Collection '{collection_name}' created successfully."

    def delete_collection(self, collection_name):
        if not self.collection_exists(collection_name):
            return f"Collection '{collection_name}' does not exist."
        
        self.client.delete_collection(collection_name=collection_name)
        return f"Collection '{collection_name}' deleted successfully."

    def load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_json_data(self, data_path):
        if os.path.isdir(data_path):
            all_data = []
            for filename in os.listdir(data_path):
                if filename.endswith('.json'):
                    with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
                        all_data.extend(json.load(f))
            return all_data
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
