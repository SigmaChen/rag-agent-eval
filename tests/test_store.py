from unittest.mock import MagicMock

from rag_agent_eval.retrieval.store import index_chunks


def test_index_chunks_replaces_existing_chunks_for_each_source():
    collection = MagicMock()
    chunks = [
        {"chunk_id": "a.md::0", "source": "a.md", "text": "A0"},
        {"chunk_id": "a.md::1", "source": "a.md", "text": "A1"},
        {"chunk_id": "b.md::0", "source": "b.md", "text": "B0"},
    ]

    count = index_chunks(chunks, collection)

    assert count == 3
    assert collection.delete.call_count == 2
    deleted_sources = {
        invocation.kwargs["where"]["source"]
        for invocation in collection.delete.call_args_list
    }
    assert deleted_sources == {"a.md", "b.md"}
    collection.upsert.assert_called_once_with(
        ids=["a.md::0", "a.md::1", "b.md::0"],
        documents=["A0", "A1", "B0"],
        metadatas=[
            {"source": "a.md"},
            {"source": "a.md"},
            {"source": "b.md"},
        ],
    )


def test_index_chunks_with_no_chunks_does_not_modify_collection():
    collection = MagicMock()

    count = index_chunks([], collection)

    assert count == 0
    collection.delete.assert_not_called()
    collection.upsert.assert_not_called()
