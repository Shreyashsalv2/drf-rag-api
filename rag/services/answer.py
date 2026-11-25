"""Retrieval and answers.

Without an API key the answer is extractive: the sentences from the retrieved chunks closest to the
question, each tagged with its source number. With GROQ_API_KEY or GEMINI_API_KEY set, an LLM writes
the answer from the same numbered sources and is told to cite them and to say when they don't cover it.
"""

import logging
import os
import re
from dataclasses import dataclass

import numpy as np
import requests
from django.conf import settings

from rag.models import Chunk

from . import vectorstore
from .embeddings import embed

log = logging.getLogger(__name__)


@dataclass
class Hit:
    chunk: Chunk
    score: float


def search(query: str, k: int = 5, document_ids: list[int] | None = None) -> list[Hit]:
    ranked = vectorstore.query(query, k, document_ids)
    chunks = Chunk.objects.select_related("document").in_bulk([chunk_id for chunk_id, _ in ranked])
    return [Hit(chunks[cid], score) for cid, score in ranked if cid in chunks]


SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+(?=[(\"'A-Z0-9])")


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 3) -> str:
    candidates: list[tuple[str, int]] = []
    for n, hit in enumerate(hits, start=1):
        for sentence in SENTENCE_END.split(hit.chunk.text.replace("\n", " ")):
            sentence = sentence.strip()
            if 25 <= len(sentence) <= 600:
                candidates.append((sentence, n))
    if not candidates:
        return "No passage in the uploaded documents answers this."
    vectors = embed([question] + [s for s, _ in candidates])
    similarity = vectors[1:] @ vectors[0]
    best = sorted(np.argsort(-similarity)[:max_sentences])  # keep reading order within the chosen ones
    return " ".join(f"{candidates[i][0]} [{candidates[i][1]}]" for i in best)


def _context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{n}] ({hit.chunk.document.title}, page {hit.chunk.page})\n{hit.chunk.text}"
        for n, hit in enumerate(hits, start=1)
    )


SYSTEM_PROMPT = (
    "Answer the question using only the numbered sources. Cite every claim with its source number in "
    "square brackets, like [2]. If the sources do not contain the answer, say so in one sentence."
)


def llm_provider() -> str | None:
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    return None


def llm_answer(question: str, hits: list[Hit]) -> str:
    provider = llm_provider()
    prompt = f"Sources:\n{_context(hits)}\n\nQuestion: {question}"
    timeout = settings.RAG["LLM_TIMEOUT"]
    if provider == "groq":
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            json={
                "model": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
                "temperature": 0,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    if provider == "gemini":
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
            json={
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    raise RuntimeError("No LLM provider configured")


def answer(question: str, k: int = 5, document_ids: list[int] | None = None) -> dict:
    hits = search(question, k, document_ids)
    result = {"question": question, "sources": hits, "note": None}
    if not hits:
        result.update(answer="There are no documents to search yet.", mode="none")
        return result
    provider = llm_provider()
    if provider:
        try:
            result.update(answer=llm_answer(question, hits), mode=f"llm:{provider}")
            return result
        except (requests.RequestException, KeyError, IndexError, ValueError) as error:
            log.warning("LLM call failed, falling back to extractive answer: %s", error)
            result["note"] = "The LLM call failed, so this is an extractive answer."
    result.update(answer=extractive_answer(question, hits), mode="extractive")
    return result
