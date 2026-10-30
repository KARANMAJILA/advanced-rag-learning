
class TextChunker:

    def __init__(
        self,
        chunk_size: int = 100,
        overlap: int = 20
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be >= 0 and smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str):
        """Split text into overlapping word-based chunks."""

        words = text.split()
        chunks = []

        step = self.chunk_size - self.overlap

        for start in range(0, len(words), step):

            chunk_words = words[
                start:start + self.chunk_size
            ]

            if not chunk_words:
                break

            chunks.append(" ".join(chunk_words))

        return chunks

    def create_chunks(self, documents):
        """Create chunks while preserving document metadata."""

        all_chunks = []

        for document in documents:

            text_chunks = self.split_text(
                document["text"]
            )

            for index, chunk_text in enumerate(text_chunks):

                all_chunks.append({
                    "chunk_id": (
                        f"{document['document_id']}_{index}"
                    ),
                    "text": chunk_text,
                    "metadata": {
                        "document_id": document["document_id"],
                        "filename": document["filename"],
                        "source": document["source"],
                        "chunk_index": index
                    }
                })

        return all_chunks
