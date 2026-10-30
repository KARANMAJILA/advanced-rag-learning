
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import TextChunker
from src.embeddings.embedder import EmbeddingGenerator

from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.vector_search import VectorRetriever


def main():

    print("\n===== ADVANCED RAG PIPELINE =====")

    # 1. Load documents
    loader = DocumentLoader("data/documents")
    documents = loader.load_txt()

    print("Documents loaded:", len(documents))

    # 2. Chunk documents
    chunker = TextChunker(
        chunk_size=100,
        overlap=20
    )

    chunks = chunker.create_chunks(documents)

    print("Chunks created:", len(chunks))

    # 3. Generate embeddings
    embedder = EmbeddingGenerator()

    embedded_chunks = embedder.generate_document_embeddings(
        chunks
    )

    # 4. Initialize vector store
    vector_store = FAISSVectorStore(
        dimension=embedder.dimension
    )

    # 5. Store embeddings
    vector_store.add_documents(embedded_chunks)

    # 6. Save vector database
    vector_store.save()

    # 7. Initialize retriever
    retriever = VectorRetriever(
        embedder,
        vector_store
    )

    # 8. Ask a question
    query = "How many annual leaves do employees receive?"

    print("\nUser Question:", query)

    results = retriever.retrieve(
        query,
        top_k=3
    )

    print("\n===== RETRIEVED DOCUMENTS =====")

    for result in results:

        print("\nChunk ID:", result["chunk_id"])
        print("Text:", result["text"])
        print("Source:", result["metadata"]["filename"])

        print(
            "Similarity:",
            round(result["similarity_score"], 4)
        )


if __name__ == "__main__":
    main()
