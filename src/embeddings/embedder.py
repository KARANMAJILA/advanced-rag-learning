
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingGenerator:

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        self.dimension = self.model.get_sentence_embedding_dimension()

        print(f"Embedding dimension: {self.dimension}")

    def generate_embeddings(self, texts):
        """Generate normalized embeddings for text."""

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32
            )

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return embeddings.astype(np.float32)

    def generate_document_embeddings(self, chunks):
        """Generate embeddings while preserving chunk metadata."""

        texts = [
            chunk["text"] for chunk in chunks
        ]

        embeddings = self.generate_embeddings(texts)

        embedded_chunks = []

        for chunk, embedding in zip(chunks, embeddings):

            embedded_chunks.append({
                **chunk,
                "embedding": embedding
            })

        return embedded_chunks
