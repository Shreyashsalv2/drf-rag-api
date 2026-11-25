"""Text embeddings with WordLlama: a small static embedding model whose weights ship inside the pip package.

No API key, no download, ~1 s to load. 256-dimensional, L2-normalised vectors (cosine = dot product).
"""

from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np

MODEL_NAME = "wordllama-l2-supercat-256"


@lru_cache(maxsize=1)
def model():
    import wordllama
    from wordllama import WordLlama

    # The package bundles the weights and tokenizer; point the loader at them so it never downloads.
    return WordLlama.load(cache_dir=Path(wordllama.__file__).parent, disable_download=True)


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 256), dtype=np.float32)
    return np.asarray(model().embed(list(texts), norm=True), dtype=np.float32)


class WordLlamaEmbedding(chromadb.EmbeddingFunction):
    """Lets ChromaDB embed documents and queries with the same model."""

    def __init__(self) -> None:
        pass

    def __call__(self, input):  # noqa: A002 (name required by ChromaDB)
        return [row.tolist() for row in embed(list(input))]

    @staticmethod
    def name() -> str:
        return MODEL_NAME

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "WordLlamaEmbedding":
        return WordLlamaEmbedding()
