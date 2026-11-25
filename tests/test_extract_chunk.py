import pytest

from rag.services.chunking import chunk_pages
from rag.services.extract import ExtractionError, clean_text, extract_pages, file_type_for
from tests.conftest import make_docx, make_pdf


class TestExtract:
    def test_file_types(self):
        assert file_type_for("Report.PDF") == "pdf"
        assert file_type_for("notes.markdown") == "md"
        with pytest.raises(ExtractionError, match="Unsupported file type"):
            file_type_for("image.png")

    def test_pdf_keeps_one_string_per_page(self):
        pages = extract_pages(make_pdf(["First page about apples.", "Second page about pears."]), "pdf")
        assert len(pages) == 2
        assert "apples" in pages[0] and "pears" in pages[1]

    def test_docx_reads_paragraphs_and_tables(self):
        pages = extract_pages(make_docx(["Hello from Word."], table=[["Name", "Role"], ["Asha", "Engineer"]]), "docx")
        assert pages == ["Hello from Word.\nName | Role\nAsha | Engineer"]

    def test_text_and_markdown(self):
        assert extract_pages("﻿Line one\r\n\r\n\r\n\r\nLine   two".encode(), "txt") == ["Line one\n\nLine two"]

    def test_pdf_without_text_is_rejected(self):
        with pytest.raises(ExtractionError, match="No text found"):
            extract_pages(make_pdf([""]), "pdf")

    def test_broken_files_are_rejected(self):
        with pytest.raises(ExtractionError):
            extract_pages(b"%PDF-1.7 this is not really a pdf", "pdf")
        with pytest.raises(ExtractionError):
            extract_pages(b"not a zip", "docx")

    def test_clean_text_rejoins_hyphenated_words(self):
        assert clean_text("data pro-\ncessing  and\tstorage ") == "data processing and storage"


class TestChunking:
    PAGES = [
        " ".join(f"Sentence {i} on page one talks about data retention and storage limits." for i in range(40)),
        "Short second page.",
    ]

    def test_chunks_respect_size_and_point_back_to_the_page(self):
        chunks = chunk_pages(self.PAGES, size=300, overlap=60)
        assert all(len(c.text) <= 300 for c in chunks)
        for c in chunks:
            page = self.PAGES[c.page - 1]
            assert page[c.start_char : c.start_char + len(c.text)] == c.text
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_chunks_never_cross_pages(self):
        chunks = chunk_pages(self.PAGES, size=300, overlap=60)
        assert chunks[-1].page == 2 and chunks[-1].text == "Short second page."
        assert {c.page for c in chunks} == {1, 2}

    def test_neighbouring_chunks_overlap(self):
        chunks = [c for c in chunk_pages(self.PAGES, size=300, overlap=80) if c.page == 1]
        for a, b in zip(chunks, chunks[1:], strict=False):
            assert b.start_char < a.start_char + len(a.text)  # b starts inside a

    def test_a_very_long_word_is_cut(self):
        chunks = chunk_pages(["x" * 950], size=400, overlap=0)
        assert [len(c.text) for c in chunks] == [400, 400, 150]

    def test_bad_settings_are_rejected(self):
        with pytest.raises(ValueError):
            chunk_pages(["text"], size=50)
        with pytest.raises(ValueError):
            chunk_pages(["text"], size=400, overlap=300)
