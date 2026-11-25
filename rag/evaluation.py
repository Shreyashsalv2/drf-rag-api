"""Retrieval evaluation: does the right passage come back in the top k?

The sample documents start every article with a heading line like "Article 17 - Right to erasure".
Each chunk is labelled with the article(s) it contains, and a question counts as answered at rank r
if the r-th retrieved chunk carries one of the question's expected articles.
Metrics: recall@k (share of questions answered in the top k) and MRR (mean of 1/rank, 0 if missed).
"""

import json
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings

from rag.services.chunking import TextChunk, chunk_pages
from rag.services.embeddings import WordLlamaEmbedding
from rag.services.extract import extract_pages, file_type_for

HEADING = re.compile(r"^Article (\d+) - ", re.MULTILINE)
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def label_chunks(chunks: list[TextChunk]) -> list[set[int]]:
    """Articles present in each chunk of one document (chunks in reading order)."""
    labels, current = [], None
    for chunk in chunks:
        found = [(m.start(), int(m.group(1))) for m in HEADING.finditer(chunk.text)]
        here = {number for _, number in found}
        # Text before the first heading belongs to the article that was open when the chunk started.
        if current is not None and (not found or found[0][0] > 20):
            here.add(current)
        labels.append(here)
        if found:
            current = found[-1][1]
    return labels


def load_sample_documents(directory: Path = SAMPLES_DIR) -> list[tuple[str, list[str]]]:
    docs = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}:
            docs.append((path.name, extract_pages(path.read_bytes(), file_type_for(path.name))))
    return docs


def load_questions(path: Path = SAMPLES_DIR / "questions.json") -> list[dict]:
    return json.loads(path.read_text())


def evaluate(docs: list[tuple[str, list[str]]], questions: list[dict], size: int, overlap: int, k: int = 5) -> dict:
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False, allow_reset=True))
    name = f"eval-{size}-{overlap}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(
        name, embedding_function=WordLlamaEmbedding(), metadata={"hnsw:space": "cosine"}
    )
    labels: dict[str, set[int]] = {}
    ids, texts = [], []
    for doc_name, pages in docs:
        chunks = chunk_pages(pages, size=size, overlap=overlap)
        for chunk, chunk_labels in zip(chunks, label_chunks(chunks), strict=True):
            chunk_id = f"{doc_name}#{chunk.index}"
            ids.append(chunk_id)
            texts.append(chunk.text)
            labels[chunk_id] = chunk_labels
    for i in range(0, len(ids), 256):
        collection.add(ids=ids[i : i + 256], documents=texts[i : i + 256])

    per_question, reciprocal_ranks = [], []
    for q in questions:
        result = collection.query(query_texts=[q["question"]], n_results=k)
        rank = next((r for r, cid in enumerate(result["ids"][0], start=1) if labels[cid] & set(q["expected"])), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        per_question.append({"question": q["question"], "expected": q["expected"], "rank": rank})
    client.delete_collection(name)
    hits = sum(1 for p in per_question if p["rank"])
    return {
        "chunk_size": size,
        "overlap": overlap,
        "chunks": len(ids),
        "k": k,
        "questions": len(questions),
        "recall_at_k": round(hits / len(questions), 3),
        "hits": hits,
        "mrr": round(sum(reciprocal_ranks) / len(questions), 3),
        "per_question": per_question,
    }
