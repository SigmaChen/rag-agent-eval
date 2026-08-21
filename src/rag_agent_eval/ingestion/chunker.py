from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: list[dict], chunk_size: int = 512, chunk_overlap: int = 50
) -> list[dict]:
    """Split documents into chunks. Each chunk keeps a reference to source path."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for i, text in enumerate(splits):
            chunks.append({
                "chunk_id": f"{doc['path']}::{i}",
                "source": doc["path"],
                "text": text,
            })
    return chunks
