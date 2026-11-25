"""ChromaDB index of chunk embeddings.

The database (Chunk rows) is the source of truth; ChromaDB is an index over it. If the index is
empty or out of step (new process, in-memory index, restored database), it is rebuilt from the table.
"""

import logging
import threading

import chromadb
from chromadb.config import Settings
from django.conf import settings

from .embeddings import WordLlamaEmbedding

log = logging.getLogger(__name__)
_lock = threading.Lock()
_client = None
_synced = False
BATCH = 256


def _get_client():
    global _client
    if _client is None:
        chroma_settings = Settings(anonymized_telemetry=False, allow_reset=True)
        path = settings.RAG["CHROMA_PATH"]
        if path:
            _client = chromadb.PersistentClient(path=path, settings=chroma_settings)
        else:
            _client = chromadb.EphemeralClient(settings=chroma_settings)
    return _client


def _collection(name: str | None = None):
    return _get_client().get_or_create_collection(
        name or settings.RAG["COLLECTION"],
        embedding_function=WordLlamaEmbedding(),
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks) -> None:
    chunks = list(chunks)
    col = _collection()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        col.add(
            ids=[c.vector_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[{"document_id": c.document_id, "page": c.page, "index": c.index} for c in batch],
        )


def delete_document(document_id: int) -> None:
    _collection().delete(where={"document_id": document_id})


def count() -> int:
    return _collection().count()


def reset() -> None:
    """Drop the index (tests, rebuilds)."""
    global _synced
    with _lock:
        try:
            _get_client().delete_collection(settings.RAG["COLLECTION"])
        except Exception:  # collection did not exist yet
            pass
        _synced = False


def ensure_synced() -> None:
    """Rebuild the index from the database once per process if the counts differ."""
    global _synced
    if _synced:
        return
    with _lock:
        if _synced:
            return
        from rag.models import Chunk

        if count() != Chunk.objects.count():
            log.info("Rebuilding the vector index from the database")
            try:
                _get_client().delete_collection(settings.RAG["COLLECTION"])
            except Exception:
                pass
            add_chunks(Chunk.objects.all().order_by("id"))
        _synced = True


def mark_synced() -> None:
    global _synced
    _synced = True


def query(text: str, k: int, document_ids: list[int] | None = None) -> list[tuple[int, float]]:
    """(chunk id, cosine similarity) for the k nearest chunks, best first."""
    ensure_synced()
    col = _collection()
    total = col.count()
    if total == 0:
        return []
    where = {"document_id": {"$in": document_ids}} if document_ids else None
    result = col.query(query_texts=[text], n_results=min(k, total), where=where, include=["distances"])
    ids, distances = result["ids"][0], result["distances"][0]
    return [(int(i.removeprefix("chunk-")), round(1 - float(d), 4)) for i, d in zip(ids, distances, strict=True)]
