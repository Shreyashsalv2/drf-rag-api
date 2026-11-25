"""Upload pipeline: file -> text per page -> chunks -> database rows -> vector index."""

import logging
from pathlib import Path

from django.conf import settings
from django.db import transaction

from rag.models import Chunk, Document

from . import vectorstore
from .chunking import chunk_pages
from .extract import extract_pages, file_type_for

log = logging.getLogger(__name__)


def ingest_bytes(filename: str, data: bytes, title: str | None = None, is_sample: bool = False) -> Document:
    file_type = file_type_for(filename)
    pages = extract_pages(data, file_type)
    pieces = chunk_pages(pages, size=settings.RAG["CHUNK_SIZE"], overlap=settings.RAG["CHUNK_OVERLAP"])
    vectorstore.ensure_synced()
    with transaction.atomic():
        document = Document.objects.create(
            title=(title or Path(filename).stem.replace("_", " ").replace("-", " ")).strip()[:255],
            filename=Path(filename).name[:255],
            file_type=file_type,
            size_bytes=len(data),
            pages=len(pages),
            is_sample=is_sample,
        )
        Chunk.objects.bulk_create(
            Chunk(document=document, index=p.index, page=p.page, start_char=p.start_char, text=p.text) for p in pieces
        )
    chunks = list(document.chunks.all())
    try:
        vectorstore.add_chunks(chunks)
    except Exception:
        log.exception("Indexing failed; removing document %s", document.pk)
        document.delete()
        raise
    return document


def remove_document(document: Document) -> None:
    vectorstore.delete_document(document.pk)
    document.delete()
