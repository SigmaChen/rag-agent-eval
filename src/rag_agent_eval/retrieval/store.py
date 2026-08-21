import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


_COLLECTION_NAME = "rag_knowledge_base"


def get_collection(
    vectordb_path: str = "data/vectordb",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=vectordb_path)
    ef = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    return client.get_or_create_collection(name=_COLLECTION_NAME, embedding_function=ef)


def index_chunks(chunks: list[dict], collection: chromadb.Collection) -> int:
    """Upsert chunks into the vector store. Returns count indexed."""
    if not chunks:
        return 0
    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return len(chunks)


def search(query: str, collection: chromadb.Collection, top_k: int = 5) -> list[dict]:
    """Return top-k relevant chunks for a query."""
    results = collection.query(query_texts=[query], n_results=top_k)
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", ""),
            "score": results["distances"][0][i] if results["distances"] else 0.0,
        })
    return hits
