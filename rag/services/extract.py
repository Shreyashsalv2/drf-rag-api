"""Turn uploaded files (PDF, DOCX, TXT, Markdown) into clean text, one string per page."""

import io
import re
from pathlib import Path

SUPPORTED_TYPES = {"pdf", "docx", "txt", "md"}


class ExtractionError(ValueError):
    """The file can't be read or contains no text."""


def file_type_for(filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix == "markdown":
        suffix = "md"
    if suffix not in SUPPORTED_TYPES:
        raise ExtractionError(f"Unsupported file type '.{suffix}'. Upload a PDF, DOCX, TXT or MD file.")
    return suffix


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ").replace("﻿", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pdf_pages(data: bytes) -> list[str]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ExtractionError("The PDF is password-protected.")
        return [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as error:
        raise ExtractionError(f"Could not read the PDF: {error}") from error


def _docx_pages(data: bytes) -> list[str]:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as error:  # python-docx raises several exception types for broken files
        raise ExtractionError("Could not read the DOCX file.") from error
    blocks = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            blocks.append(" | ".join(cell.text for cell in row.cells))
    return ["\n".join(blocks)]  # DOCX has no fixed pages; the whole file counts as page 1


def extract_pages(data: bytes, file_type: str) -> list[str]:
    """Cleaned text per page. Raises ExtractionError if nothing readable is found."""
    if file_type == "pdf":
        raw = _pdf_pages(data)
    elif file_type == "docx":
        raw = _docx_pages(data)
    elif file_type in {"txt", "md"}:
        raw = [data.decode("utf-8", errors="replace")]
    else:
        raise ExtractionError(f"Unsupported file type '{file_type}'.")
    pages = [clean_text(p) for p in raw]
    if not any(pages):
        raise ExtractionError("No text found. Scanned PDFs (images only) are not supported.")
    return pages
