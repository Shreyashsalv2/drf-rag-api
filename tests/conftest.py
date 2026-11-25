import io

import pytest
from rest_framework.test import APIClient

from rag.services import vectorstore


@pytest.fixture(autouse=True)
def clean_index():
    """Each test starts with an empty vector index (the database is rolled back by pytest-django)."""
    vectorstore.reset()
    yield
    vectorstore.reset()


@pytest.fixture(autouse=True)
def fresh_throttle_counts():
    from django.core.cache import cache

    cache.clear()
    yield
    cache.clear()


@pytest.fixture(autouse=True)
def no_llm_keys(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def api():
    return APIClient()


def make_pdf(pages: list[str]) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_font("Helvetica", size=11)
    for text in pages:
        pdf.add_page()
        if text:
            pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def make_docx(paragraphs: list[str], table: list[list[str]] | None = None) -> bytes:
    import docx

    document = docx.Document()
    for p in paragraphs:
        document.add_paragraph(p)
    if table:
        t = document.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                t.cell(r, c).text = value
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


BREACH_TEXT = (
    "Personal data breach notification. In the case of a personal data breach, the controller shall notify "
    "the supervisory authority without undue delay and, where feasible, not later than 72 hours after "
    "having become aware of it. The notification shall describe the nature of the breach."
)
CATS_TEXT = (
    "Caring for a cat. Cats need fresh water, a balanced diet and a clean litter box. Play with your cat "
    "every day and take it to the vet once a year for vaccinations."
)
