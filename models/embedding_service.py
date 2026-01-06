import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import tqdm
import pickle
import os

class EmbeddingService:
    def __init__(self, dense_model_name="all-MiniLM-L6-v2"):
        print(f"Loading dense model: {dense_model_name}...")
        self.dense_model = SentenceTransformer(dense_model_name)
        self.tfidf_vectorizer = TfidfVectorizer(lowercase=True)

    def get_dense_embeddings(self, texts, prefix=""):
        """Generates dense vectors for a list of strings with optional prefix."""
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]
        return self.dense_model.encode(texts, show_progress_bar=False).tolist()

    def fit_sparse_model(self, texts):
        """Fits the TF-IDF model on the corpus to establish vocabulary."""
        print("Fitting sparse model on corpus...")
        self.tfidf_vectorizer.fit(texts)

    def save_sparse_model(self, path):
        """Saves the fitted TF-IDF model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        print(f"Sparse model saved to {path}")

    def load_sparse_model(self, path):
        """Loads a fitted TF-IDF model from disk."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            print(f"Sparse model loaded from {path}")
        else:
            print(f"Warning: Sparse model path {path} not found.")

    def get_sparse_embeddings(self, texts):
        """Generates sparse vectors (indices and values) for Qdrant."""
        tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        sparse_results = []
        
        # Get feature names for mapping or just use indices
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix.getrow(i)
            indices = row.indices.tolist()
            values = row.data.tolist()
            sparse_results.append({"indices": indices, "values": values})
            
        return sparse_results

    @staticmethod
    def flatten_content(doc, keys):
        """Extracts and lowercases content from specified keys in a dict."""
        values = []
        for key in keys:
            val = doc.get(key, "")
            if isinstance(val, list):
                values.extend([str(v).lower() for v in val])
            else:
                values.append(str(val).lower())
        return " ".join(values)
