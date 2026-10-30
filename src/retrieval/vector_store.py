
import faiss
import numpy as np
import pickle
from pathlib import Path


class FAISSVectorStore:

    def __init__(self, dimension, storage_path="data/vector_store"):

        self.dimension = dimension
        self.storage_path = Path(storage_path)

        self.storage_path.mkdir(
            parents=True,
            exist_ok=True
        )

        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

    def add_documents(self, embedded_chunks):
        """Store vectors and their corresponding metadata."""

        if not embedded_chunks:
            return

        embeddings = np.array(
            [item["embedding"] for item in embedded_chunks],
            dtype=np.float32
        )

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError("Invalid embedding dimensions")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain invalid values")

        self.index.add(embeddings)

        for item in embedded_chunks:

            self.documents.append({
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "metadata": item["metadata"]
            })

        print(f"Stored {len(embedded_chunks)} vectors")

    def search(self, query_embedding, top_k=3):
        """Retrieve the most similar document chunks."""

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError("Query embedding dimension mismatch")

        if not np.isfinite(query_embedding).all():
            raise ValueError("Invalid query embedding")

        if not np.isclose(
            np.linalg.norm(query_embedding),
            1.0,
            atol=1e-3
        ):
            raise ValueError("Query embedding must be normalized")

        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx < 0:
                continue

            document = self.documents[idx]

            results.append({
                "chunk_id": document["chunk_id"],
                "text": document["text"],
                "metadata": document["metadata"],
                "similarity_score": float(score)
            })

        return results

    def save(self):
        """Persist FAISS index and metadata."""

        faiss.write_index(
            self.index,
            str(self.storage_path / "index.faiss")
        )

        with open(
            self.storage_path / "documents.pkl",
            "wb"
        ) as file:

            pickle.dump(self.documents, file)

        print("Vector store saved successfully")

    def load(self):
        """Load persisted index and metadata."""

        index_path = self.storage_path / "index.faiss"
        metadata_path = self.storage_path / "documents.pkl"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Vector store files not found")

        loaded_index = faiss.read_index(str(index_path))

        with open(metadata_path, "rb") as file:
            documents = pickle.load(file)

        if loaded_index.d != self.dimension:
            raise ValueError("Stored index dimension mismatch")

        if loaded_index.ntotal != len(documents):
            raise ValueError("Index and metadata are inconsistent")

        self.index = loaded_index
        self.documents = documents

        print(f"Loaded {self.index.ntotal} vectors")
